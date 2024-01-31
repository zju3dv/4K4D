import torch
import mcubes
import trimesh
import numpy as np

from typing import Callable, Tuple, Union

from tqdm import tqdm
from functools import reduce
from pytorch3d.structures import Meshes

from pytorch3d.ops.laplacian_matrices import laplacian, cot_laplacian, norm_laplacian

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.bound_utils import get_bounds
from easyvolcap.utils.net_utils import take_gradient
from easyvolcap.utils.chunk_utils import linear_gather, multi_gather, multi_gather_tris
from easyvolcap.utils.math_utils import normalize, torch_unique_with_indices_and_inverse
from easyvolcap.utils.sample_utils import get_voxel_grid_and_update_bounds, sample_closest_points

from torch.autograd.function import FunctionCtx, Function, once_differentiable


def unmerge_faces(faces: torch.Tensor, *args):
    # stack into pairs of (vertex index, texture index)
    stackable = [faces.reshape(-1)]
    # append multiple args to the correlated stack
    # this is usually UV coordinates (vt) and normals (vn)
    for arg in args:
        stackable.append(arg.reshape(-1))

    # unify them into rows of a numpy array
    stack = torch.column_stack(stackable)
    # find unique pairs: we're trying to avoid merging
    # vertices that have the same position but different
    # texture coordinates
    _, unique, inverse = torch_unique_with_indices_and_inverse(stack)

    # only take the unique pairs
    pairs = stack[unique]
    # try to maintain original vertex order
    order = pairs[:, 0].argsort()
    # apply the order to the pairs
    pairs = pairs[order]

    # we re-ordered the vertices to try to maintain
    # the original vertex order as much as possible
    # so to reconstruct the faces we need to remap
    remap = torch.zeros(len(order), dtype=torch.long, device=faces.device)
    remap[order] = torch.arange(len(order), device=faces.device)

    # the faces are just the inverse with the new order
    new_faces = remap[inverse].reshape((-1, 3))

    # the mask for vertices and masks for other args
    result = [new_faces]
    result.extend(pairs.T)

    return result


def merge_faces(faces, *args, n_verts=None):
    # TODO: batch this
    # remember device the faces are on
    device = faces.device
    # start with not altering faces at all
    result = [faces]
    # find the maximum index referenced by faces
    if n_verts is None:  # sometimes things get padded
        n_verts = faces.max() + 1
    # add a vertex mask which is just ordered
    result.append(torch.arange(n_verts, device=device))

    # now given the order is fixed do our best on the rest of the order
    for arg in args:
        # create a mask of the attribute-vertex mapping
        # note that these might conflict since we're not unmerging
        masks = torch.zeros((3, n_verts), dtype=torch.long, device=device)
        # set the mask using the unmodified face indexes
        for i, f, a in zip(range(3), faces.permute(*torch.arange(faces.ndim - 1, -1, -1)), arg.permute(*torch.arange(arg.ndim - 1, -1, -1))):
            masks[i][f] = a
        # find the most commonly occurring attribute (i.e. UV coordinate)
        # and use that index note that this is doing a float conversion
        # and then median before converting back to int: could also do this as
        # a column diff and sort but this seemed easier and is fast enough
        result.append(torch.median(masks, dim=0)[0].to(torch.long))

    return result


def meshes_attri_laplacian_smoothing(meshes: Meshes, attri: torch.Tensor, method: str = "uniform"):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    attri_packed = attri.reshape(-1, attri.shape[-1])

    if method == "uniform":
        loss = L.mm(attri_packed)
    elif method == "cot":
        loss = L.mm(attri_packed) * norm_w - attri_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(attri_packed) - L_sum * attri_packed) * norm_w
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum() / N


class RegisterSDFGradient(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, verts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], chunk_size=65536):
        ctx.save_for_backward(verts)
        ctx.decoder = decoder
        ctx.chunk_size = chunk_size
        return verts

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor):
        chunk_size = ctx.chunk_size
        decoder: Callable[[torch.Tensor], torch.Tensor] = ctx.decoder
        verts: torch.Tensor = ctx.saved_tensors[0]

        verts = verts.detach().requires_grad_()  # should not affect the original verts
        with torch.enable_grad():
            sdf = torch.cat([decoder(verts[i:i + chunk_size]) for i in range(0, verts.shape[0], chunk_size)])
            norm = normalize(take_gradient(sdf, verts, create_graph=False, retain_graph=True))  # N, 3
            grad = -torch.einsum('ni,ni->n', norm, grad) * sdf.view(verts.shape[0])  # N
            loss = grad.sum()
        loss.backward(retain_graph=True)  # accumulate gradients into decorder parameters
        return None, None, None


register_sdf_gradient = RegisterSDFGradient.apply


def differentiable_marching_cubes(points: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], chunk_size=65536):
    """
    Will use torchmcubes and return the corrsponding vertices of the marching cubes result
    currently no batch dimension supported

    TODO: use octree to make this faster

    points: [X, Y, Z]
    """

    sh = points.shape
    points = points.view(-1, 3)
    upper = points.max(dim=0, keepdim=True)[0]
    lower = points.min(dim=0, keepdim=True)[0]
    points = points.detach().requires_grad_(False)  # should not affect the original verts
    with torch.no_grad():
        sdf = np.concatenate(
            [
                decoder(points[i:i + chunk_size]).detach().to('cpu').numpy()
                for i in range(0, points.shape[0], chunk_size)
            ]
        )
        # MARK: GPU CPU SYNC
    # verts, faces = marching_cubes(-sdf.view(*sh[:-1]), 0.0)
    verts, faces = mcubes.marching_cubes(-sdf.reshape(*sh[:-1]), 0.0)
    verts = torch.from_numpy(verts.astype(np.float32)).to(points.device, non_blocking=True)
    faces = torch.from_numpy(faces.astype(np.int32)).to(points.device, non_blocking=True)
    verts = verts * (upper - lower) / (torch.tensor(sh[:-1], device=points.device) - 1) + lower

    verts = register_sdf_gradient(verts, decoder, chunk_size)  # literally a state switch, no other stuff
    return verts, faces


def face_normals(v: torch.Tensor, f: torch.Tensor):
    # compute faces normals w.r.t the vertices (considering batch dimension)
    tris = multi_gather_tris(v, f)

    # Compute face normals
    v0, v1, v2 = torch.split(tris, split_size_or_sections=1, dim=-2)
    v0, v1, v2 = v0[..., 0, :], v1[..., 0, :], v2[..., 0, :]
    e1 = v1 - v0
    e2 = v2 - v1
    face_normals = torch.cross(e1, e2)

    face_normals = normalize(face_normals)

    return face_normals


def get_face_connectivity(faces: torch.Tensor):
    '''
    Returns a list of adjacent face pairs
    '''

    he = triangle_to_halfedge(None, faces)
    twin = he.twin
    vert = he.vert
    edge = he.edge
    hedg = torch.arange(he.HE, device=faces.device)  # HE, :record half edge indices
    face = hedg // 3
    manifold = (twin >= 0).nonzero(as_tuple=True)[0]

    # NOTE: some repeated computation
    edge_manifold = edge[manifold]  # manifold edge indices

    args = edge_manifold.argsort()  # 00, 11, 22 ...
    inds = hedg[manifold][args]  # index of the valid half edges

    connected_faces = face[inds].view(-1, 2)
    edges_connecting_faces = vert[inds].view(-1, 2)

    return edges_connecting_faces, connected_faces


def get_edge_length(v, e):
    v0 = linear_gather(v, e[..., 0], dim=-2)
    v1 = linear_gather(v, e[..., 1], dim=-2)
    return torch.norm(v0 - v1, dim=-1)


def get_vertex_mass(v: torch.Tensor, f: torch.Tensor, density: float, areas=None):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    if areas is None:
        areas = get_face_areas(v, f)
    triangle_masses = density * areas

    vertex_masses = torch.zeros(v.shape[:-1], device=v.device, dtype=v.dtype)
    vertex_masses[f[..., 0]] += triangle_masses / 3
    vertex_masses[f[..., 1]] += triangle_masses / 3
    vertex_masses[f[..., 2]] += triangle_masses / 3

    return vertex_masses


def get_face_areas(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    v0 = v[f[..., 0]]
    v1 = v[f[..., 1]]
    v2 = v[f[..., 2]]

    u = v2 - v0
    v = v1 - v0

    return torch.norm(torch.cross(u, v), dim=-1) / 2.0


def loop_subdivision(v: torch.Tensor, f: torch.Tensor, steps=2):
    halfedge = triangle_to_halfedge(v, f)
    halfedge = multiple_halfedge_loop_subdivision(halfedge, steps)
    return halfedge_to_triangle(halfedge)


def segment_mesh(verts: torch.Tensor,
                 faces: torch.Tensor,
                 vs: torch.Tensor,
                 inds: torch.Tensor,
                 smoothing: str = 'mesh',
                 dilate=0,
                 ):
    # prepare vertex semantics
    vs_bits = 1 << vs

    # prepare indices of semantics to preserve
    inds_bits = (1 << inds).sum()  # no repeatition

    # prepare faces to be preserved
    vm = (inds_bits & vs_bits) != 0

    # dilate the vertex mask along
    if dilate < 0:
        vm = ~vm
    vm = vm.float()
    edges, i, count = get_edges(faces)
    A = adjacency(verts, edges)
    for i in range(abs(dilate)):
        vm = A @ vm
    vm = vm.bool()
    if dilate < 0:
        vm = ~vm

    # extract face mask
    tm = multi_gather_tris(vm, faces, dim=-1)  # F, 3
    fm = tm.sum(dim=-1) != 0

    # create the extracted mesh
    f, vm = unmerge_faces(faces[fm])
    v = verts[vm]

    # perform laplacian smoothing on edge vertices / faces (or maybe just the whole mesh to acquire a better shape?)
    e, i, c = get_edges(f)
    if smoothing == 'edge':
        svi = e[c != 2].ravel().unique()  # selected vertices' indices
    else:
        svi = None
    v = laplacian_smoothing(v, e, svi)

    # Fill single triangle and single quad holes in the current mesh. Will remove non-manifold vertices maybe?
    mesh = trimesh.Trimesh(v.detach().cpu().numpy(), f.detach().cpu().numpy())
    mesh.fill_holes()
    v, f = mesh.vertices, mesh.faces

    # Convert back to torch.Tensor
    v, f = torch.tensor(v, device=verts.device, dtype=torch.float), torch.tensor(f, device=verts.device, dtype=torch.long)

    return v, f


def icp_loss(v: torch.Tensor,  # input points
             t: torch.Tensor,  # triangles
             n0: torch.Tensor,  # normal of the input points: v
             n1: torch.Tensor,  # normal of the target triangles: t
             dist_th: float = 0.1,  # 10cm
             angle_th: float = np.cos(np.deg2rad(45.0)),
             ) -> torch.Tensor:

    # we don't expect batch dimension here
    v, t, n0, n1 = v[None], t[None], n0[None], n1[None]

    from bvh_distance_queries import BVH

    bvh = BVH()  # NOTE: wasteful!
    dists_sq, points, face_ids, barys = bvh(t, v)  # forward distance, find closest point on tris_padded of every smplhs vertices
    delta = v - points

    # use distance and angle fileter to get a good estimation
    filter0 = dists_sq < dist_th ** 2  # distance to the closest surface should be within range
    n1 = multi_gather(n1, face_ids)
    direction = (n1 * n0).sum(dim=-1)
    filter1 = direction > angle_th  # angle between tar_src surface normal should be within range
    filter = filter0 & filter1

    # compute actual l2 loss
    loss = (delta[filter] ** 2).sum(dim=-1).mean()  # L2 loss

    return loss


def bidirectional_icp_fitting(v0: torch.Tensor,
                              f0: torch.Tensor,
                              v1: torch.Tensor,
                              f1: torch.Tensor,
                              lambda_smooth: int = 29,
                              opt_iter: int = 500,
                              ep_iter: int = 50,
                              lr: float = 3e-2,
                              boundary_focus: bool = True,
                              dilate: int = 0,
                              ):
    """
    Robust Bidirectional Mesh Fitting
    TODO: Need to investigate why the meshes got stretched along the orthogonal of the normal direction
    """

    if boundary_focus:
        # select vertices to focus optimization on
        e0, i0, c0 = get_edges(f0)
        e1, i1, c1 = get_edges(f1)
        svi0 = e0[c0 != 2].ravel().unique()  # selected vertices' indices: boundary
        svi1 = e1[c1 != 2].ravel().unique()  # selected vertices' indices: boundary

        # dialte the boundary edge selection
        vm0 = torch.zeros(v0.shape[:-1], device=v0.device, dtype=torch.bool)
        vm1 = torch.zeros(v1.shape[:-1], device=v1.device, dtype=torch.bool)
        vm0[svi0] = True
        vm1[svi1] = True
        A0 = adjacency(v0, e0)
        A1 = adjacency(v1, e1)
        vm0 = vm0.float()
        vm1 = vm1.float()
        for i in range(abs(dilate)):
            vm0 = A0 @ vm0
            vm1 = A1 @ vm1
        vm0 = vm0.bool()
        vm1 = vm1.bool()
        svi0 = vm0.nonzero(as_tuple=True)[0]
        svi1 = vm1.nonzero(as_tuple=True)[0]
    else:
        svi0, svi1 = None, None

    from largesteps.optimize import AdamUniform
    from largesteps.geometry import compute_matrix
    from largesteps.parameterize import from_differential, to_differential

    # assume no batch dim
    M0 = compute_matrix(v0, f0, lambda_smooth)
    M1 = compute_matrix(v1, f1, lambda_smooth)
    p0 = to_differential(M0, v0)
    p1 = to_differential(M1, v1)
    p0.requires_grad_()
    p1.requires_grad_()
    optim = AdamUniform([p0, p1], lr=lr)

    pbar = tqdm(range(opt_iter))
    for i in range(opt_iter):
        v0 = from_differential(M0, p0, 'Cholesky')
        v1 = from_differential(M1, p1, 'Cholesky')
        t0 = multi_gather_tris(v0, f0)
        t1 = multi_gather_tris(v1, f1)

        m = Meshes([v0, v1], [f0, f1])
        nv0, nv1 = m.verts_normals_list()
        nf0, nf1 = m.faces_normals_list()

        if svi0 is not None:
            v0, nv0 = v0[svi0], nv0[svi0]
        if svi1 is not None:
            v1, nv1 = v1[svi1], nv1[svi1]

        loss0 = icp_loss(v0, t1, nv0, nf1)
        loss1 = icp_loss(v1, t0, nv1, nf0)
        loss = loss0 + loss1

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        pbar.update(1)
        if i % ep_iter == 0:
            pbar.write(f'bidirectional L2 loss: {loss.item():.5g}')

    v0 = from_differential(M0, p0.detach(), 'Cholesky')
    v1 = from_differential(M1, p1.detach(), 'Cholesky')
    return v0, v1


def halfedge_loop_subdivision(halfedge: dotdict[str, torch.Tensor], is_manifold=False):
    # Please just watch this: https://www.youtube.com/watch?v=mxk2HHk1NK4
    # Adapted from https://github.com/kvanhoey/ParallelHalfedgeSubdivision
    # assuming the mesh has clean topology? except for boundary edges
    # assume no boundary edge for now!

    # loading from dotdict
    verts: torch.Tensor = halfedge.verts  # V, 3
    twin: torch.Tensor = halfedge.twin  # HE,
    vert: torch.Tensor = halfedge.vert  # HE,
    edge: torch.Tensor = halfedge.edge  # HE,

    HE = halfedge.HE
    E = halfedge.E
    F = halfedge.F
    V = halfedge.V

    NHE = 4 * HE
    NF = 4 * F
    NE = 2 * E + 3 * F
    NV = V + E

    # assign empty memory
    ntwin = torch.empty(NHE, device=vert.device, dtype=vert.dtype)
    nvert = torch.empty(NHE, device=vert.device, dtype=vert.dtype)
    nedge = torch.empty(NHE, device=vert.device, dtype=vert.dtype)

    # prepare input for topology computation
    hedg = torch.arange(HE, device=vert.device)
    next = hedg - (hedg % 3) + (hedg + 1) % 3
    prev = hedg - (hedg % 3) + (hedg + 2) % 3

    next_twin = next[twin]
    twin_prev = twin[prev]
    edge_prev = edge[prev]

    # assign next topology
    i0 = 3 * hedg + 0
    i1 = 3 * hedg + 1
    i2 = 3 * hedg + 2
    i3 = 3 * HE + hedg

    ntwin[i0] = 3 * next_twin + 2
    ntwin[i1] = 3 * HE + hedg
    ntwin[i2] = 3 * twin_prev
    ntwin[i3] = 3 * hedg + 1

    nedge[i0] = 2 * edge + (hedg < twin).long()
    nedge[i1] = 2 * E + hedg
    nedge[i2] = 2 * edge_prev + (prev > twin_prev).long()
    nedge[i3] = nedge[i1]

    nvert[i0] = vert
    nvert[i1] = V + edge
    nvert[i2] = V + edge_prev
    nvert[i3] = nvert[i2]

    if not is_manifold:
        # deal with non-manifold cases
        manifold_mask = twin >= 0
        non_manifold_mask = ~manifold_mask
        non_manifold_mask_prev = non_manifold_mask[prev]
        manifold = manifold_mask.nonzero(as_tuple=True)[0]  # only non manifold half edge are here
        non_manifold = non_manifold_mask.nonzero(as_tuple=True)[0]  # only non manifold half edge are here
        non_manifold_prev = non_manifold_mask_prev.nonzero(as_tuple=True)[0]  # only non manifold half edge are here

        ntwin[i0[non_manifold]] = twin[non_manifold]
        ntwin[i2[non_manifold_prev]] = twin_prev[non_manifold_prev]  # should store the non-manifold twin (whether previsou is non-manifold)
        nedge[i0[non_manifold]] = 2 * edge[non_manifold]
        nedge[i2[non_manifold_prev]] = 2 * edge_prev[non_manifold_prev] + 1  # should store the non-manifold edge (whether previous is non-manifold)

    from torch_scatter import scatter

    # pre-compute vertex velance & beta values
    _, inverse, velance = vert.unique(sorted=False, return_inverse=True, return_counts=True)
    velance = scatter(velance[inverse], vert, dim=0, dim_size=V, reduce='mean')  # all verts velance, in original order (not some sorted order)
    beta = (1 / velance) * (5 / 8 - (3 / 8 + 1 / 4 * torch.cos(2 * torch.pi / velance))**2)

    # prepare geometric topological variables
    vert_next = vert[next]
    vert_prev = vert[prev]
    vert_edge = V + edge  # no duplication between vert and vert_edge
    velance_vert = velance[vert]
    beta_vert = beta[vert]

    if not is_manifold:
        # prepare for computing non-manifold original vertices
        incident = -twin[non_manifold]
        non_manifold_vert_mask = scatter(non_manifold_mask.int(), inverse, dim=0, dim_size=V, reduce='max').bool()  # non_manifold vert mask, sorted
        non_manifold_vert_mask = non_manifold_vert_mask[inverse]  # non-manifold edge mask, if vert is non-manifold, this would be non-manifold
        non_manifold_vert = non_manifold_vert_mask.nonzero(as_tuple=True)[0]

        # prev and current is manifold
        manifold_prev = manifold_mask[prev].nonzero(as_tuple=True)[0]
        non_manifold_prev_twin_prev_mask = torch.zeros(HE, device=vert.device, dtype=manifold_mask.dtype)
        non_manifold_prev_twin_prev_mask[manifold_prev] = non_manifold_mask[prev[twin_prev[manifold_prev]]]
        non_manifold_prev_twin_prev = non_manifold_prev_twin_prev_mask.nonzero(as_tuple=True)[0]

    # actually distribute geometric values, if no topology change, only these should be retained
    verts_vert = verts[vert]
    verts_vert_next = verts[vert_next]
    verts_vert_prev = verts[vert_prev]
    nverts_vert_edge = (3 * verts_vert + verts_vert_prev) / 8  # vertex position for vertices created on edges
    nverts_vert = (1 / velance_vert - beta_vert)[..., None] * verts_vert + beta_vert[..., None] * verts_vert_next  # vertex position for older vertices

    if not is_manifold:
        # non-manifold edges
        nverts_vert_edge[non_manifold] = (verts_vert[non_manifold] + verts_vert_next[non_manifold]) / (incident * 2)[..., None]
        nverts_vert[non_manifold_vert] = 0
        nverts_vert[non_manifold] += 1 / 8 * verts_vert_next[non_manifold] + 3 / 8 * verts_vert[non_manifold]
        nverts_vert[non_manifold_prev_twin_prev] += 1 / 8 * verts_vert_prev[twin_prev[non_manifold_prev_twin_prev]] + 3 / 8 * verts_vert[twin_prev[non_manifold_prev_twin_prev]]

    # origianl vertices and new edge vertices will have no overlap
    nverts_vert_edge = scatter(nverts_vert_edge, vert_edge, dim=0, dim_size=NV)
    nverts_vert = scatter(nverts_vert, vert, dim=0, dim_size=NV)
    nverts = nverts_vert_edge + nverts_vert

    # prepare return dotdict structure
    nhalfedge = dotdict()

    # geometric info
    nhalfedge.verts = nverts  # V, 3

    # topologinal info
    nhalfedge.twin = ntwin  # HE,
    nhalfedge.vert = nvert  # HE,
    nhalfedge.edge = nedge  # HE,

    # size info (some of them could be omitted like HE)
    nhalfedge.HE = NHE
    nhalfedge.E = NE
    nhalfedge.F = NF
    nhalfedge.V = NV

    return nhalfedge


def multiple_halfedge_loop_subdivision(halfedge: dotdict[str, torch.Tensor], steps=2, is_manifold=False):
    for i in range(steps):
        halfedge = halfedge_loop_subdivision(halfedge, is_manifold)
    return halfedge


def halfedge_to_triangle(halfedge: dotdict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # assuming the mesh has clean topology? except for boundary edges
    # assume no boundary edge for now!
    verts = halfedge.verts
    vert = halfedge.vert  # HE,

    HE = len(vert)
    hedg = torch.arange(HE, device=verts.device)
    next = hedg & ~3 | (hedg + 1) & 3

    e = torch.stack([vert, vert[next]], dim=-1)
    e01, e12, e20 = e[0::3], e[1::3], e[2::3]
    faces = torch.stack([e01[..., 0], e12[..., 0], e20[..., 0]], dim=-1)

    return verts, faces


def triangle_to_halfedge(verts: Union[torch.Tensor, None],
                         faces: torch.Tensor,
                         is_manifold: bool = False,
                         ):
    # assuming the mesh has clean topology? except for boundary edges
    # assume no boundary edge for now!
    F = len(faces)
    V = len(verts) if verts is not None else faces.max().item()
    HE = 3 * F

    # create halfedges
    v0, v1, v2 = faces.chunk(3, dim=-1)
    e01 = torch.cat([v0, v1], dim=-1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=-1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=-1)  # (sum(F_n), 2)

    # stores the vertex indices for each half edge
    e = torch.empty(HE, 2, device=faces.device, dtype=faces.dtype)
    e[0::3] = e01
    e[1::3] = e12
    e[2::3] = e20
    vert = e[..., 0]  # HE, :record starting half edge
    vert_next = e[..., 1]

    edges = torch.stack([torch.minimum(vert_next, vert), torch.maximum(vert_next, vert)], dim=-1)
    hash = V * edges[..., 0] + edges[..., 1]  # HE, 2, contains edge hash, should be unique
    _, edge, counts = hash.unique(sorted=False, return_inverse=True, return_counts=True)
    E = len(counts)

    hedg = torch.arange(HE, device=faces.device)  # HE, :record half edge indices

    if is_manifold:
        inds = edge.argsort()  # 00, 11, 22 ...
        twin = torch.empty_like(inds)
        twin[inds[0::2]] = inds[1::2]
        twin[inds[1::2]] = inds[0::2]

    else:
        # now we have edge indices, if it's a good mesh, each edge should have two half edges
        # in some non-manifold cases this would be broken so we need to first filter those non-manifold edges out
        manifold = counts == 2  # non-manifold mask
        manifold = manifold[edge]  # non-manifold half edge mask

        edge_manifold = edge[manifold]  # manifold edge indices

        args = edge_manifold.argsort()  # 00, 11, 22 ...
        inds = hedg[manifold][args]
        twin_manifold = torch.empty_like(inds)
        twin_manifold[args[0::2]] = inds[1::2]
        twin_manifold[args[1::2]] = inds[0::2]

        twin = torch.empty(HE, device=faces.device, dtype=torch.long)
        twin[manifold] = twin_manifold
        twin[~manifold] = -counts[edge][~manifold]  # non-manifold half edge mask, number of half edges stored in the twin

    # should return these values
    halfedge = dotdict()

    # geometric info
    halfedge.verts = verts  # V, 3

    # connectivity info
    halfedge.twin = twin  # HE,
    halfedge.vert = vert  # HE,
    halfedge.edge = edge  # HE,

    halfedge.HE = HE
    halfedge.E = E
    halfedge.F = F
    halfedge.V = V

    return halfedge


def winding_number(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Parallel implementation of the Generalized Winding Number of points on the mesh
    O(n_points * n_faces) memory usage, parallelized execution

    1. Project tris onto the unit sphere around every points
    2. Compute the signed solid angle of the each triangle for each point
    3. Sum the solid angle of each triangle

    Parameters
    ----------
    pts    : torch.Tensor, (n_points, 3)
    verts  : torch.Tensor, (n_verts, 3)
    faces  : torch.Tensor, (n_faces, 3)

    This implementation is also able to take a/multiple batch dimension
    """
    # projection onto unit sphere: verts implementation gives a little bit more performance
    uv = verts[..., None, :, :] - pts[..., :, None, :]  # n_points, n_verts, 3
    uv = uv / uv.norm(dim=-1, keepdim=True)  # n_points, n_verts, 3

    # gather from the computed vertices (will result in a copy for sure)
    expanded_faces = faces[..., None, :, :].expand(*faces.shape[:-2], pts.shape[-2], *faces.shape[-2:])  # n_points, n_faces, 3

    u0 = multi_gather(uv, expanded_faces[..., 0])  # n, f, 3
    u1 = multi_gather(uv, expanded_faces[..., 1])  # n, f, 3
    u2 = multi_gather(uv, expanded_faces[..., 2])  # n, f, 3

    e0 = u1 - u0  # n, f, 3
    e1 = u2 - u1  # n, f, 3
    del u1

    # compute solid angle signs
    sign = (torch.cross(e0, e1) * u2).sum(dim=-1).sign()

    e2 = u0 - u2
    del u0, u2

    l0 = e0.norm(dim=-1)
    del e0

    l1 = e1.norm(dim=-1)
    del e1

    l2 = e2.norm(dim=-1)
    del e2

    # compute edge lengths: pure triangle
    l = torch.stack([l0, l1, l2], dim=-1)  # n_points, n_faces, 3

    # compute spherical edge lengths
    l = 2 * (l / 2).arcsin()  # n_points, n_faces, 3

    # compute solid angle: preparing: n_points, n_faces
    s = l.sum(dim=-1) / 2
    s0 = s - l[..., 0]
    s1 = s - l[..., 1]
    s2 = s - l[..., 2]

    # compute solid angle: and generalized winding number: n_points, n_faces
    eps = 1e-10  # NOTE: will cause nan if not bigger than 1e-10
    solid = 4 * (((s / 2).tan() * (s0 / 2).tan() * (s1 / 2).tan() * (s2 / 2).tan()).abs() + eps).sqrt().arctan()
    signed_solid = solid * sign  # n_points, n_faces

    winding = signed_solid.sum(dim=-1) / (4 * torch.pi)  # n_points

    return winding


winding_number.constant = 72  # 3 * 3 * 4: 2, reduced from summed up number to 2, totally 6 N, F, 3 tensors existing


def ray_stabbing(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, multiplier: int = 1):
    """
    Check whether a bunch of points is inside the mesh defined by verts and faces
    effectively calculating their occupancy values

    Parameters
    ----------
    ray_o : torch.Tensor(float), (n_rays, 3)
    verts : torch.Tensor(float), (n_verts, 3)
    faces : torch.Tensor(long), (n_faces, 3)
    """
    n_rays = pts.shape[0]
    pts = pts[None].expand(multiplier, n_rays, -1)
    pts = pts.reshape(-1, 3)
    ray_d = torch.rand_like(pts)  # (n_rays, 3)
    ray_d = normalize(ray_d)  # (n_rays, 3)
    u, v, t = moller_trumbore(pts, ray_d, multi_gather_tris(verts, faces))  # (n_rays, n_faces, 3)
    inside = ((t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()  # (n_rays, n_faces)
    inside = (inside.count_nonzero(dim=-1) % 2).bool()  # if mod 2 is 0, even, outside, inside is odd
    inside = inside.view(multiplier, n_rays, -1)
    inside = inside.sum(dim=0) / multiplier  # any show inside mesh
    return inside


def moller_trumbore(ray_o: torch.Tensor, ray_d: torch.Tensor, tris: torch.Tensor, eps=1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The Moller Trumbore algorithm for fast ray triangle intersection
    Naive batch implementation (m rays and n triangles at the same time)
    O(n_rays * n_faces) memory usage, parallelized execution

    Parameters
    ----------
    ray_o : torch.Tensor, (n_batch, n_rays, 3)
    ray_d : torch.Tensor, (n_batch, n_rays, 3)
    tris  : torch.Tensor, (n_batch, n_faces, 3, 3)
    """
    E1 = tris[..., 1, :] - tris[..., 0, :]  # vector of edge 1 on triangle (n_faces, 3)
    E2 = tris[..., 2, :] - tris[..., 0, :]  # vector of edge 2 on triangle (n_faces, 3)

    # batch cross product
    N = torch.cross(E1, E2)  # normal to E1 and E2, automatically batched to (n_faces, 3)

    # invdet = 1. / -(torch.einsum('md,nd->mn', ray_d, N) + eps)  # inverse determinant (n_faces, 3)
    invdet = 1. / -((ray_d[..., :, None, :] * N[..., None, :, :]).sum(dim=-1) + eps)  # inverse determinant (n_faces, 3)

    A0 = ray_o[..., :, None, :] - tris[..., None, :, 0, :]  # (n_rays, 3) - (n_faces, 3) -> (n_rays, n_faces, 3) automatic broadcast
    DA0 = torch.cross(A0, ray_d[..., :, None, :].expand(*A0.shape))  # (n_rays, n_faces, 3) x (n_rays, 3) -> (n_rays, n_faces, 3) no automatic broadcast

    u = (DA0 * E2[..., None, :, :]).sum(dim=-1) * invdet
    v = -(DA0 * E1[..., None, :, :]).sum(dim=-1) * invdet
    t = (A0 * N[..., None, :, :]).sum(dim=-1) * invdet  # t >= 0.0 means this is a ray

    return u, v, t


def winding_distance(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, winding_th=0.45):
    from bvh_distance_queries import BVH

    winding = winding_number(pts, verts, faces)  # let's say its gonna be 0 or 1
    bvh = BVH()  # NOTE: wasteful!
    p: torch.Tensor = bvh(multi_gather_tris(verts[None], faces[None]), pts[None])[1][0, ...]  # remove last dimension
    d = (pts - p).norm(dim=-1)
    d = d

    winding_shift = ((winding - 0.5) * 2).clip(-1, 1)
    winding_shift[winding_shift < (-1 + winding_th)] = -1
    winding_shift[winding_shift > (+1 - winding_th)] = +1
    winding_d = winding_shift * d

    return winding_d


def bvh_distance(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor):
    from bvh_distance_queries import BVH

    bvh = BVH()  # NOTE: wasteful!
    p: torch.Tensor = bvh(multi_gather_tris(verts[None], faces[None]), pts[None])[1][0, ...]  # remove last dimension
    d = (pts - p).norm(dim=-1)
    return d


def hierarchical_winding_distance_remesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    init_voxel_size=0.05,  # 5cm voxels
    init_dist_th_verts=1.0,  # 50cm hole range
    init_dist_th_tris=0.25,  # 50cm hole range
    steps=4,
    **kwargs,
):
    guide_verts, guide_faces = verts, faces
    voxel_size, dist_th_verts, dist_th_tris = init_voxel_size, init_dist_th_verts, init_dist_th_tris
    decay = np.power(dist_th_tris / (voxel_size / 2**(steps - 2)), 1 / (steps - 1)) if steps > 1 else -1
    # log(decay)
    for i in range(int(steps)):
        guide_verts, guide_faces = winding_distance_remesh(verts, faces, guide_verts, guide_faces, voxel_size, dist_th_verts, dist_th_tris, **kwargs)
        voxel_size, dist_th_verts, dist_th_tris = voxel_size / 2, dist_th_verts / decay, dist_th_tris / decay

    return guide_verts, guide_faces


def winding_number_nooom(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, quota_GB=15.0):
    # allocate chunk size to avoid oom when computing winding number
    faces_cnt_shape = faces.shape[:-1]
    faces_cnt = reduce(lambda x, y: x * y, faces_cnt_shape)
    quota_B = quota_GB * 2 ** 30  # GB -> B
    chunk = int(quota_B / (faces_cnt * winding_number.constant))  # quota = tris_cnt * pts_cnt * winding_number.constant

    # compute winding_number_distance on GPU and store results on CPU
    winding = []
    for i in tqdm(range(0, pts.shape[-2], chunk)):
        pts_chunk = pts[..., i:i + chunk, :]
        winding_chunk = winding_number(pts_chunk, verts, faces)
        winding.append(winding_chunk)
    winding = torch.cat(winding, dim=-1)

    return winding


def winding_distance_remesh(verts: torch.Tensor,
                            faces: torch.Tensor,

                            guide_verts: torch.Tensor = None,
                            guide_faces: torch.Tensor = None,

                            voxel_size=0.005,  # 5mm voxel size
                            dist_th_verts=0.05,  # 5cm range
                            dist_th_tris=0.01,  # 1cm range

                            quota_GB=15.0,  # GB of VRAM
                            level_set=0.5,  # where to segment for the winding number
                            winding_th=0.75,  # 0.45 range to filter unnecessary winding number
                            ):
    """
    Robust Inside-Outside Segmentation using Generalized Winding Numbers
    https://www.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.html
    Naive GPU parallel implementation of the described algorithm with distance guidance

    Note that we formulate the segmentation problem as a simple remesh problem
    dist_th_verts should be no smaller than the maximum of edge lengths and hole lengths
    dist_th_tris should be no smaller than the maximum hole lengths
    """

    log(f'voxel_size: {voxel_size}')
    log(f'dist_th_verts: {dist_th_verts}')
    log(f'dist_th_tris: {dist_th_tris}')

    if guide_verts is None or guide_faces is None:
        guide_verts, guide_faces = verts, faces

    # NOTE: requires fake batch dimension
    wbounds = get_bounds(verts[None], dist_th_tris)
    pts, wbounds = get_voxel_grid_and_update_bounds([voxel_size, voxel_size, voxel_size], wbounds)  # B, N, 3
    sh = pts.shape[1:-1]
    pts = pts.view(-1, 3)  # P, 3
    wbounds = wbounds[0]  # remove batch

    # level 1 filtering: based on vertex distance: KNN with K == 1
    d0 = sample_closest_points(pts[None], guide_verts[None])[0, ..., 0]
    d1 = sample_closest_points(pts[None], verts[None])[0, ..., 0]
    close_verts = torch.minimum(d0, d1) < dist_th_verts
    pts = pts[close_verts]

    # level 2 filtering: distance to the surface point (much faster than pytorch3d impl)
    d0 = bvh_distance(pts, guide_verts, guide_faces)
    d1 = bvh_distance(pts, verts, faces)
    close_tris = torch.minimum(d0, d1) < dist_th_tris
    pts = pts[close_tris]
    d = d1[close_tris]

    winding = winding_number_nooom(pts, verts, faces, quota_GB)

    winding_shift = ((winding - level_set) * 2).clip(-1, 1)
    winding_shift[winding_shift < (-1 + winding_th)] = -1
    winding_shift[winding_shift > (+1 - winding_th)] = +1

    winding_d = winding_shift * d  # winding distance

    # possibly visualize the queried winding_distance
    # rgb = colormap(torch.tensor(winding, device=verts.device) / 2 * 100 + 0.5)
    # export_pynt_pts(pts, rgb, filename='winding.ply')

    # undo two levels of filtering
    close_tris = close_tris
    cube_tris = torch.ones(close_tris.shape, dtype=torch.float, device=verts.device) * -10
    cube_tris[close_tris] = winding_d
    close_verts = close_verts
    cube_verts = torch.ones(close_verts.shape, dtype=torch.float, device=verts.device) * -10
    cube_verts[close_verts] = cube_tris
    cube = cube_verts.view(*sh)

    # perform marching cubes to extract mesh (linear interpolation is actually good enought if we use winding_distance instead of winding_number)
    v, f = mcubes.marching_cubes(cube.detach().cpu().numpy(), 0.0)
    v = v.astype(np.float32)
    f = f.astype(np.int64)

    # we assume the inside surface is always smaller thant the outside surface
    mesh = trimesh.Trimesh(v, f)
    mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.vertices))  # get largest component (removing floating artifacts automatically)
    v = mesh.vertices
    f = mesh.faces

    # fix marching cube result size
    v *= voxel_size
    v += wbounds[0].detach().cpu().numpy()

    v = torch.tensor(v, device=verts.device, dtype=verts.dtype)
    f = torch.tensor(f, device=verts.device, dtype=faces.dtype)

    return v, f


def get_edges(faces: torch.Tensor):
    V = faces.max()
    F = faces.shape[0]
    HE = F * 3

    # create halfedges
    v0, v1, v2 = faces.chunk(3, dim=-1)
    e01 = torch.cat([v0, v1], dim=-1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=-1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=-1)  # (sum(F_n), 2)

    # stores the vertex indices for each half edge
    e = torch.empty(HE, 2, device=faces.device, dtype=faces.dtype)
    e[0::3] = e01
    e[1::3] = e12
    e[2::3] = e20
    vert = e[..., 0]  # HE, :record starting half edge
    vert_next = e[..., 1]

    edges = torch.stack([torch.minimum(vert_next, vert), torch.maximum(vert_next, vert)], dim=-1)
    hash = V * edges[..., 0] + edges[..., 1]  # HE, 2, contains edge hash, should be unique
    u, i, c = hash.unique(sorted=False, return_inverse=True, return_counts=True)

    e = torch.stack([u // V, u % V], dim=1)
    return e, i, c


def adjacency(verts: torch.Tensor, edges: torch.Tensor):
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse.FloatTensor(idx, ones, (V, V))
    return A


def laplacian(verts: torch.Tensor, edges: torch.Tensor):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L


def laplacian_smoothing(v: torch.Tensor, e: torch.Tensor, inds: torch.Tensor = None, alpha=0.33, iter=90):
    for i in range(iter):
        # 1st gaussian smoothing pass
        L = laplacian(v, e)
        vln = L @ v
        if inds is None:
            v += alpha * vln
        else:
            v[inds] += alpha * vln[inds]

        # 2nd gaussian smoothing pass
        L = laplacian(v, e)
        vln = L @ v
        if inds is None:
            v += -(alpha + 0.01) * vln
        else:
            v[inds] += -(alpha + 0.01) * vln[inds]
    return v


def isosurface_fitting(src_verts: torch.Tensor,  # assume no batch dim
                       src_faces: torch.Tensor,
                       tar_verts: torch.Tensor,
                       tar_faces: torch.Tensor,
                       f: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = winding_distance,  # might not be optimizable on well defined surface...
                       l: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x, y: (x - y).pow(2).sum(),
                       param_lambda: int = 29,
                       thresh: float = 0.5,  # iso surface thresh
                       opt_iter: int = 100,
                       chunk_size: int = 1600,
                       lr: float = 1e-3,
                       ):

    from largesteps.optimize import AdamUniform
    from largesteps.geometry import compute_matrix
    from largesteps.parameterize import from_differential, to_differential

    M = compute_matrix(src_verts, src_faces, param_lambda)
    param = to_differential(M, src_verts)
    param.requires_grad_()
    optim = AdamUniform([param], lr=lr)

    p = tqdm(range(opt_iter))
    for i in range(opt_iter):
        verts = from_differential(M, param, 'Cholesky')
        query = linear_gather(verts, torch.randperm(len(verts), device=verts.device)[:chunk_size])
        level = f(
            query,
            tar_verts,
            tar_faces,
        )
        loss = l(level, thresh)  # L2 loss on winding number
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        p.update(1)
        p.desc = f'l2: {loss.item():.6f}'

    verts = from_differential(M, param.detach(), 'Cholesky')
    return verts


def average_edge_length(verts, faces, backend='pytorch'):
    """
    Compute the average length of all edges in a given mesh.

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions of [N, 3]
    faces : torch.Tensor
        array of triangle faces of [F, 3]
    """
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    if backend == 'pytorch':
        norm_func = partial(torch.norm, dim=-1)
    elif backend == 'np':
        norm_func = partial(np.linalg.norm, axis=-1)
    else:
        return NotImplementedError
    A = norm_func(v1 - v2)
    B = norm_func(v0 - v2)
    C = norm_func(v0 - v1)

    return (A + B + C).sum() / faces.shape[0] / 3


avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
max_pool_3d = torch.nn.MaxPool3d(3, stride=1, padding=1)


@torch.no_grad()
def get_surface_sliding(
    sdf,
    resolution=256,
    bounding_box_min=[-1.0, -1.0, -1.0],
    bounding_box_max=[1.0, 1.0, 1.0],
    return_mesh=False,
    level=0,
    coarse_mask=None,
    output_path="test.ply",
    simplify_mesh=False,
    targetfacenum=500000,
    replace=False,
):
    from skimage import measure
    assert resolution % 256 == 0
    if coarse_mask is not None:
        coarse_mask = coarse_mask.permute(2, 1, 0)[None, None].cuda().float()

    resN = resolution
    cropN = 256
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max

    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # print(xs)
    # print(ys)
    # print(zs)
    pbar = tqdm(total=N**3)
    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf(pnts[None])[0])
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                if coarse_mask is not None:
                    # breakpoint()
                    points_tmp = points.permute(1, 2, 3, 0)[None].cuda()
                    current_mask = torch.nn.functional.grid_sample(coarse_mask, points_tmp)
                    current_mask = (current_mask > 0.0).cpu().numpy()[0, 0]
                else:
                    current_mask = None

                points_pyramid = [points]
                for _ in range(3):
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]

                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min) / cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                    if mask is None:
                        # only evaluate
                        if coarse_mask is not None:
                            pts_sdf = torch.ones_like(pts[:, 1])
                            valid_mask = (
                                torch.nn.functional.grid_sample(coarse_mask, pts[None, None, None])[0, 0, 0, 0] > 0
                            )
                            if valid_mask.any():
                                pts_sdf[valid_mask] = evaluate(pts[valid_mask].contiguous())
                        else:
                            pts_sdf = evaluate(pts)
                    else:
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]

                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        # print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.0

                z = pts_sdf.detach().cpu().numpy()

                # skip if no surface found
                if current_mask is not None:
                    valid_z = z.reshape(cropN, cropN, cropN)[current_mask]
                    if valid_z.shape[0] <= 0 or (np.min(valid_z) > level or np.max(valid_z) < level):
                        continue

                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),  # .transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                        mask=current_mask,
                    )
                    # print(np.array([x_min, y_min, z_min]))
                    # print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])
                    # print(verts.min(), verts.max())

                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    # meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)
                pbar.update()

    combined = trimesh.util.concatenate(meshes)

    if return_mesh:
        return combined
    else:
        filename = str(output_path)
        if replace:
            filename_simplify = str(output_path)
        else:
            filename_simplify = str(output_path).replace(".ply", "-simplify.ply")
        combined.merge_vertices(digits_vertex=6)
        combined.export(filename)
        if simplify_mesh:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(filename)

            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=targetfacenum)
            ms.save_current_mesh(filename_simplify, save_face_color=False)
