import torch
import xatlas
import numpy as np
from tqdm import tqdm
from torch_scatter import scatter_add, scatter_mean

import sys

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.mesh_utils import loop_subdivision
from easyvolcap.utils.net_utils import take_gradient, take_jacobian
from easyvolcap.utils.data_utils import export_dotdict, load_mesh, export_mesh, to_numpy
from easyvolcap.utils.prof_utils import setup_profiler, profiler_start, profiler_step, profiler_stop
from easyvolcap.utils.physx_utils import Garment, StVKMaterial, bending_energy_constraints, stretch_energy_constraints, bending_energy, stretch_energy


def main(
    time=1/60,  # 1s for the simulation (physics time)
    steps=360,  # 5s for the simulation (physics steps
    substeps=100,  # 1ms for simulation substep

    omega=1.0,

    # opt_iter * lr -> how compliant the mesh is
    bending_compliance=1e-9,
    stretch_compliance=0.0,
    opt_iter=1,
    gravity=[0.0, -9.8, 0.0],  # m/s**2
    input_file='tshirt.obj',
    output_file='tshirt-drop#{}.npz',
    device='cuda',
):
    gravity = torch.tensor(gravity, device=device)
    delta_t = time / substeps

    v, f, vm, fm = load_mesh(input_file, load_uv=True, device=device)
    # v, f = loop_subdivision(v, f, 1)
    # vm, fm = loop_subdivision(vm, fm, 1)

    # # perform automatic uv unwrapping
    # vmap, fm, vm = xatlas.parametrize(v.detach().cpu().numpy(), f.detach().cpu().numpy())
    # vmap, fm, vm = vmap.astype(np.int64), fm.astype(np.int64), vm.astype(np.float32)
    # vmap, fm, vm = torch.tensor(vmap, device=v.device), torch.tensor(fm, device=v.device), torch.tensor(vm, device=v.device)

    # construct the actual garment
    position = v
    velocity = torch.zeros_like(v)
    material = StVKMaterial()
    garment = Garment(v, f, vm, fm, material)

    position_145 = position[145]
    position_224 = position[224]

    profiler_start()
    animation = []
    pbar = tqdm(total=steps * substeps * opt_iter)
    for i in range(steps):
        tqdm.write(f'\nTimestep: {(i+1) * time:.3f}')
        for j in range(substeps):
            velocity = velocity + delta_t * gravity
            previous = position
            position = position + delta_t * velocity

            stretch_lambda = torch.zeros(len(garment.f), dtype=torch.float, device=position.device)  # F
            bending_lambda = torch.zeros(len(garment.f_connectivity_edges), dtype=torch.float, device=position.device)  # E

            # this part can also be considered as an optimization problem
            # solve constraints: with conditional gradient descent for optimization purpose
            for k in range(opt_iter):
                position[145] = position_145
                # position[224] = position_224

                # position: B, V
                # gradient: B, V, F
                # will return totally B, F + B, E number of constraints

                # energy_stretch = stretch(position)  # F, 3
                # energy_bending = bending(position)  # E, 3

                # if we take the gradient, we'd get the sum of the gradient on all participated constraints here
                # thus we'd need to weight these according to the participating parts instead of the whole thing
                B = 1
                V = len(position)
                position = position[None]
                stretch_lambda = stretch_lambda[None]
                bending_lambda = bending_lambda[None]

                energy_stretch, grad, grad_d2, delta_lambda, delta_p = stretch_energy_constraints(position, garment, delta_t=delta_t, accum_lambda=stretch_lambda, compliance=stretch_compliance)  # B, F, 3, 3
                delta_position = scatter_mean(delta_p.view(-1, 3), garment.f[None].view(-1), dim=0, dim_size=B * V)  # B * F * 3, 3 & B * F * 3
                stretch_lambda = stretch_lambda + delta_lambda * omega
                position = position + delta_position * omega
                position[0, 145] = position_145
                # position[0, 224] = position_224

                energy_bending, grad, grad_d2, delta_lambda, delta_p = bending_energy_constraints(position, garment, delta_t=delta_t, accum_lambda=bending_lambda, compliance=bending_compliance)  # B, E, 2, 3, 3
                delta_position = scatter_mean(delta_p.view(-1, 3), garment.f_connected_faces[None].view(-1), dim=0, dim_size=B * V)  # (B, E, 2, 3), 3 & (B, E, 2, 3)
                bending_lambda = bending_lambda + delta_lambda * omega
                position = position + delta_position * omega

                position = position[0]
                stretch_lambda = stretch_lambda[0]
                bending_lambda = bending_lambda[0]

                pbar.update(1)
                profiler_step()

            # tqdm.write(f'Loss: {l.item()}')
            position[145] = position_145
            # position[224] = position_224

            residual = position - previous
            velocity = residual / delta_t

        tqdm.write(f'Mean residual: {residual.norm(dim=-1).mean().item()}')
        tqdm.write(f'Max residual: {residual.norm(dim=-1).max().item()}')
        tqdm.write(f'Mean velocity: {velocity.norm(dim=-1).mean().item()}')
        tqdm.write(f'Max velocity: {velocity.norm(dim=-1).max().item()}')
        tqdm.write(f'Bending energy: {energy_bending.sum().item() / material.bending_multiplier}')
        tqdm.write(f'Stretching energy: {energy_stretch.sum().item() / material.stretch_multiplier}')

        animation.append(to_numpy(position))

    profiler_stop()
    animation = np.stack(animation)
    animated_mesh = dotdict()
    animated_mesh.animation = animation
    animated_mesh.faces = to_numpy(f)
    export_dotdict(animated_mesh, filename=output_file.format('global').replace('.ply', '.npz'))


if __name__ == '__main__':
    profiling = False

    prof_cfg = dotdict()
    prof_cfg.enabled = profiling
    prof_cfg.record_dir = 'data/record/global_method'
    prof_cfg.clear_previous = True
    prof_cfg.skip_first = 10
    prof_cfg.wait = 5
    prof_cfg.warmup = 5
    prof_cfg.active = 20
    prof_cfg.repeat = 1
    setup_profiler(prof_cfg)

    # totally 40
    steps = 2
    quality_step = 0.5

    if profiling:
        main(steps=steps, quality_step=quality_step)
    else:
        main()
