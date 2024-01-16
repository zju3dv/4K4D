from typing import Callable
import torch
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import take_gradient
from easyvolcap.utils.chunk_utils import multi_gather_tris, expand0, linear_gather
from easyvolcap.utils.math_utils import torch_inverse_2x2, torch_inverse_3x3, torch_trace, normalize
from easyvolcap.utils.mesh_utils import triangle_to_halfedge, face_normals, get_edge_length, get_face_connectivity, get_vertex_mass, get_face_areas


class StVKMaterial(dotdict):
    '''
    This class stores parameters for the StVK material model
    copied from https://github.com/isantesteban/snug/blob/main/losses/material.py

    # Fabric material parameters
    thickness = 0.00047 # (m)
    bulk_density = 426  # (kg / m3)
    area_density = thickness * bulk_density

    material = Material(
        density=area_density, # Fabric density (kg / m2)
        thickness=thickness,  # Fabric thickness (m)
        young_modulus=0.7e5, 
        poisson_ratio=0.485,
        stretch_multiplier=1,
        bending_multiplier=50
    )

    '''

    def __init__(self,
                 density=426 * 0.00047,  # Fabric density (kg / m2)
                 thickness=0.00047,  # Fabric thickness (m)
                 young_modulus=0.7e5,
                 poisson_ratio=0.485,
                 bending_multiplier=50.0,
                 stretch_multiplier=1.0,
                 material_multiplier=1.0,
                 ):

        self.density = density
        self.thickness = thickness
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

        self.bending_multiplier = bending_multiplier
        self.stretch_multiplier = stretch_multiplier

        # Bending and stretching coefficients (ARCSim)
        self.A = young_modulus / (1.0 - poisson_ratio**2)
        self.stretch_coeff = self.A
        self.stretch_coeff *= stretch_multiplier * material_multiplier

        self.bending_coeff = self.A / 12.0 * (thickness ** 3)
        self.bending_coeff *= bending_multiplier * material_multiplier

        # Lamé coefficients
        self.lame_mu = 0.5 * self.stretch_coeff * (1.0 - self.poisson_ratio)
        self.lame_lambda = self.stretch_coeff * self.poisson_ratio


class Garment(dotdict):
    '''
    This class stores mesh and material information of the garment
    No batch dimension here
    '''

    def __init__(self, v: torch.Tensor, f: torch.Tensor, vm: torch.Tensor, fm: torch.Tensor, material: StVKMaterial = StVKMaterial()):
        self.material = material

        # Face attributes
        self.f = f
        self.f_connectivity_edges, self.f_connectivity = get_face_connectivity(f)  # Pairs of adjacent faces
        self.f_connected_faces = linear_gather(self.f, self.f_connectivity.view(-1), dim=-2).view(-1, 2, 3)  # E * 2, 3 -> E, 2, 3
        self.f_area = get_face_areas(v, f)

        # Vertex attributes
        self.v = v
        self.v_mass = get_vertex_mass(v, f, self.material.density, self.f_area)
        self.inv_mass = 1 / self.v_mass

        # Rest state of the cloth (computed in material space)
        self.vm = vm
        self.fm = fm
        tris_m = multi_gather_tris(vm, fm)
        # tris_m = multi_gather_tris(v, f)
        self.Dm = get_shape_matrix(tris_m)
        self.Dm_inv = torch_inverse_2x2(self.Dm)

# batch dimension applicable


def get_shape_matrix(tris: torch.Tensor):
    return torch.stack([tris[..., 0, :] - tris[..., 2, :],
                        tris[..., 1, :] - tris[..., 2, :],
                        ], dim=-1)


def deformation_gradient(tris: torch.Tensor, Dm_invs: torch.Tensor):
    Ds = get_shape_matrix(tris)
    return Ds @ Dm_invs


def get_matching_identity_matrix(F: torch.Tensor):
    # match shape of F
    shape = F.shape[:-2]
    I = torch.eye(F.shape[-1], dtype=F.dtype, device=F.device)
    for i in shape:
        I = I[None]
    I = I.expand(*shape, -1, -1)

    return I


def green_strain_tensor(F: torch.Tensor) -> torch.Tensor:
    I = get_matching_identity_matrix(F)
    return 0.5 * (F.mT @ F - I)


def stretch_energy_constraints(v: torch.Tensor, garment: Garment, **kwargs):
    '''
    v: B, V, 3

    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant–Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''
    # XPBD step
    triangles = multi_gather_tris(v, garment.f)  # B, F, 3, 3
    inv_mass = multi_gather_tris(garment.inv_mass[None, ..., None], garment.f)[..., 0]  # B, F, 3

    def func(triangles, garment): return stretch_energy_components(triangles, garment)
    return xpbd_constraints(func, triangles, inv_mass, garment, **kwargs)


def bending_energy_constraints(v: torch.Tensor, garment: Garment, **kwargs):
    # XPBD step
    triangles = multi_gather_tris(v, garment.f_connected_faces.view(-1, 3)).view(v.shape[0], -1, 2 * 3, 3)  # B, E, 2 * 3, 3
    inv_mass = multi_gather_tris(garment.inv_mass[None, ..., None], garment.f_connected_faces.view(-1, 3))[..., 0].view(v.shape[0], -1, 2 * 3)  # B, E, 2 * 3

    def func(triangles, garment): return bending_energy_components(v, triangles.view(v.shape[0], -1, 2, 3, 3), garment)
    return xpbd_constraints(func, triangles, inv_mass, garment, **kwargs)


def xpbd_constraints(func: Callable[..., torch.Tensor], triangles: torch.Tensor, inv_mass: torch.Tensor, garment: Garment, accum_lambda: torch.Tensor = None, delta_t: float = 1, compliance: float = 0):
    triangles.requires_grad_()
    with torch.enable_grad():
        energy = func(triangles, garment)  # B, E
    grad = take_gradient(energy.sum() / energy.shape[0], triangles, create_graph=False, retain_graph=False)  # B, E, 2, 3, 3, gradient on all participating triangles
    grad = grad.detach()
    energy = energy.detach()

    grad_d2 = (grad ** 2).sum(dim=-1)  # B, E, 2, 3

    # preparation for lambda
    if accum_lambda is None:
        accum_lambda = torch.zeros_like(inv_mass)

    compliance_factor = compliance / delta_t ** 2
    delta_lambda = (-energy - compliance_factor * accum_lambda) / ((inv_mass * grad_d2).sum(dim=-1) + compliance_factor)
    delta_p = delta_lambda[..., None, None] * inv_mass[..., None] * grad  # B, F, 3, 3

    return energy, grad, grad_d2, delta_lambda, delta_p


def stretch_energy_components(triangles: torch.Tensor, garment: Garment):
    '''
    triangles: B, F, 3, 3

    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant–Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''
    B = triangles.shape[0]

    Dm_invs = expand0(garment.Dm_inv, B)  # B, F, 2, 2

    F = deformation_gradient(triangles, Dm_invs)  # B, F, 3, 2
    G = green_strain_tensor(F)  # B, F, 2, 2

    # Energy
    mat = garment.material

    I = get_matching_identity_matrix(G)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * torch_trace(G)[..., None, None] * I  # B, F, 2, 2

    energy_density = torch_trace(S.mT @ G)  # B, F
    energy = garment.f_area[None] * mat.thickness * energy_density  # B, F

    # return torch.sum(energy) / B
    return energy


def bending_energy_components(v: torch.Tensor, connected_triangles: torch.Tensor, garment: Garment) -> torch.Tensor:
    '''
    connected_triangles: B, E, 4, 3 (0, 1, 2, 3) -> edge verts and point verts?
                       : B, E, 2, 3, 3 -> triangles but no edge information?

    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    B = connected_triangles.shape[0]

    # Compute face normals
    # fn = face_normals(v, garment.f)  # B, F, 3
    # n0 = linear_gather(fn, garment.f_connectivity[:, 0], dim=-2)  # B, E, 3
    # n1 = linear_gather(fn, garment.f_connectivity[:, 1], dim=-2)  # B, E, 3
    n = torch.cross(connected_triangles[:, :, :, 1, :] - connected_triangles[:, :, :, 0, :], connected_triangles[:, :, :, 2, :] - connected_triangles[:, :, :, 1, :])
    n0 = n[:, :, 0]
    n1 = n[:, :, 1]

    # Compute edge length
    v0 = linear_gather(v, garment.f_connectivity_edges[:, 0], dim=-2)  # B, E, 3
    v1 = linear_gather(v, garment.f_connectivity_edges[:, 1], dim=-2)  # B, E, 3
    e = v1 - v0
    l = e.norm(dim=-1, keepdim=True)
    # e_norm = e / l

    # Compute area
    f_area = expand0(garment.f_area, B)
    a0 = linear_gather(f_area, garment.f_connectivity[:, 0], dim=-1)
    a1 = linear_gather(f_area, garment.f_connectivity[:, 1], dim=-1)
    a = a0 + a1

    # Compute dihedral angle between faces
    cos = (n0 * n1).sum(dim=-1)  # dot product, B, E
    crs = torch.cross(n0, n1)
    sin = (crs ** 2).sum(dim=-1) / crs.norm(dim=-1)  # B, E
    theta = torch.atan2(sin, cos)  # B, E

    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = garment.material
    scale = l[..., 0]**2 / (4 * a)

    # Bending energy
    energy = mat.bending_coeff * scale * (theta ** 2) / 2  # B, E

    return energy


def stretch_energy(v: torch.Tensor, garment: Garment):
    '''
    v: B, V, 3

    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant–Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''
    B = v.shape[0]
    triangles = multi_gather_tris(v, garment.f)  # B, F, 3, 3

    Dm_invs = expand0(garment.Dm_inv, B)  # B, F, 2, 2

    F = deformation_gradient(triangles, Dm_invs)  # B, F, 3, 2
    G = green_strain_tensor(F)  # B, F, 2, 2

    # Energy
    mat = garment.material

    I = get_matching_identity_matrix(G)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * torch_trace(G)[..., None, None] * I  # B, F, 2, 2

    energy_density = torch_trace(S.mT @ G)  # B, F
    energy = garment.f_area[None] * mat.thickness * energy_density  # B, F

    return torch.sum(energy) / B


def bending_energy(v: torch.Tensor, garment: Garment):
    '''
    v: B, V, 3

    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    B = v.shape[0]

    # Compute face normals
    fn = face_normals(v, garment.f)
    n0 = linear_gather(fn, garment.f_connectivity[:, 0], dim=-2)
    n1 = linear_gather(fn, garment.f_connectivity[:, 1], dim=-2)

    # Compute edge length
    v0 = linear_gather(v, garment.f_connectivity_edges[:, 0], dim=-2)
    v1 = linear_gather(v, garment.f_connectivity_edges[:, 1], dim=-2)
    e = v1 - v0
    l = e.norm(dim=-1, keepdim=True)
    e_norm = e / l

    # Compute area
    f_area = expand0(garment.f_area, B)
    a0 = linear_gather(f_area, garment.f_connectivity[:, 0], dim=-1)
    a1 = linear_gather(f_area, garment.f_connectivity[:, 1], dim=-1)
    a = a0 + a1

    # Compute dihedral angle between faces
    cos = (n0 * n1).sum(dim=-1)  # dot product, B, E
    sin = (e_norm * torch.cross(n0, n1)).sum(dim=-1)  # B, E
    theta = torch.atan2(sin, cos)  # B, E

    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = garment.material
    scale = l[..., 0]**2 / (4 * a)

    # Bending energy
    energy = mat.bending_coeff * scale * (theta ** 2) / 2  # B, E

    return torch.sum(energy) / B


def gravity_energy(x: torch.Tensor, mass: torch.Tensor, g=9.81):
    # S, V, 3
    # U = m * g * h
    U = g * mass * (x[:, :, 1] + 0)  # mgh

    return torch.sum(U) / x.shape[0]


def inertia_term(x_next: torch.Tensor,
                 x_curr: torch.Tensor,
                 v_curr: torch.Tensor,
                 mass: torch.Tensor,
                 delta_t
                 ):

    x_pred = x_curr + delta_t * v_curr
    x_diff = x_next - x_pred

    inertia = (x_diff ** 2).sum(dim=-1) * mass / (2 * delta_t ** 2)

    return inertia.sum() / x_next.shape[0]


def dynamic_term(x_next: torch.Tensor,
                 x_curr: torch.Tensor,
                 v_curr: torch.Tensor,
                 mass: torch.Tensor,
                 delta_t,
                 gravitational_acceleration=[0, -9.81, 0]):

    gravitational_acceleration = torch.tensor(gravitational_acceleration, device=x_next.device, dtype=x_next.dtype)

    x_pred = x_curr + delta_t * v_curr + delta_t * delta_t * gravitational_acceleration  # gravity added at last dimension for every point and timestep

    x_diff = x_next - x_pred
    dynamic = (x_diff ** 2).sum(dim=-1) * mass / (2 * delta_t ** 2)

    return dynamic.sum() / x_next.shape[0]


def inertia_term_sequence(x: torch.Tensor,
                          mass: torch.Tensor,
                          delta_t: float,
                          method: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] = inertia_term,
                          compliance: float = 1.0,
                          ):
    """
    x: torch.Tensor of shape [batch_size, num_frames, num_vertices, 3]
    """
    B = x.shape[0]
    V = x.shape[-2]
    # Compute velocities
    x_next = x[:, 1:]  # B, T-1, V, 3
    x_curr = x[:, :-1]  # B, T-1, V, 3
    v_next = (x_next - x_curr) / delta_t
    zeros = torch.zeros([B, 1, V, 3], dtype=x.dtype, device=x.device)
    v_curr = torch.cat([zeros, v_next[:, :-1]], dim=1)  # B, T-1, V, 3

    # Flatten
    x_next = x_next.view(-1, V, 3)  # B * T-1, V, 3
    x_curr = x_curr.view(-1, V, 3)  # B * T-1, V, 3, should have been the correct x_curr: for 3 frame case, this holds
    v_curr = v_curr.view(-1, V, 3)  # B * T-1, V, 3, should have been the gt v_curr: for 3 frame case, this holds

    return compliance * method(x_next, x_curr, v_curr, mass, delta_t)
