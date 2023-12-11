import torch
from tqdm import tqdm
from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix
from largesteps.parameterize import from_differential, to_differential

# fmt: off
import sys
sys.path.append('.')

from easyvolcap.utils.data_utils import load_mesh, export_mesh
from easyvolcap.utils.mesh_utils import triangle_to_halfedge, halfedge_to_triangle, multiple_halfedge_loop_subdivision
# fmt: on


def forward(p: torch.Tensor, f: torch.Tensor, M: torch.sparse.FloatTensor, depth: int):
    # this shows that our loop subdivision is differentiable w.r.t verts
    # and we can trivially connect it to the largesteps mesh optimization program
    v = from_differential(M, p, 'Cholesky')
    he = triangle_to_halfedge(v, f, True)
    nhe = multiple_halfedge_loop_subdivision(he, depth, True)
    v, f = halfedge_to_triangle(nhe)
    return v, f


def main():
    lr = 3e-2
    depth = 2
    ep_iter = 10
    opt_iter = 50
    lambda_smooth = 29
    input_file = 'big-sigcat.ply'
    output_file = 'big-sigcat-to-sphere.ply'
    v, f = load_mesh(input_file)
    he = triangle_to_halfedge(v, f, True)
    print(f'vert count: {he.V}')
    print(f'face count: {he.F}')
    print(f'edge count: {he.E}')
    print(f'halfedge count: {he.HE}')

    # assume no batch dim
    M = compute_matrix(v, f, lambda_smooth)
    p = to_differential(M, v)
    p.requires_grad_()
    optim = AdamUniform([p], lr=lr)

    print()
    pbar = tqdm(range(opt_iter))
    for i in range(opt_iter):
        v, _ = forward(p, f, M, depth)

        loss = ((v.norm(dim=-1) - 1) ** 2).sum()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        pbar.update(1)
        if i % ep_iter == 0:
            pbar.write(f'L2 loss: {loss.item():.5g}')

    v, f = forward(p.detach(), f, M, depth)
    export_mesh(v, f, filename=output_file)


if __name__ == "__main__":
    main()
