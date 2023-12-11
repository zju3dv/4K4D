import torch
import xatlas
import numpy as np
from tqdm import tqdm
from torch.optim import SGD, Adam
from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix
from largesteps.parameterize import from_differential, to_differential

# fmt: off
import sys


sys.path.append('.')

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.mesh_utils import loop_subdivision
from easyvolcap.utils.data_utils import export_dotdict, load_mesh, export_mesh, to_numpy
from easyvolcap.utils.prof_utils import setup_profiler, profiler_start, profiler_step, profiler_stop
from easyvolcap.utils.physx_utils import Garment, StVKMaterial, stretch_energy, bending_energy
# fmt: on


def main(
    time=1/60,  # 1s for the simulation (physics time)
    steps=360,  # 5s for the simulation (physics steps
    substeps=1,  # 1ms for simulation substep
    quality_step=1.0,

    # opt_iter * lr -> how compliant the mesh is
    relaxation=1.0,
    lr=4e-2,
    opt_iter=80,  # the number of iterations for optimization leads to compliance: i.e. stiffness of the simulation
    lambda_smooth=1,  # will make the object much silky or stiff
    gravity=[0.0, -9.8, 0.0],  # m/s**2
    input_file='tshirt.obj',
    output_file='tshirt-drop#{}.npz',
    device='cuda',
):
    gravity = torch.tensor(gravity, device=device)
    opt_iter = int(opt_iter * quality_step)
    lr = lr / quality_step
    delta_t = time / substeps

    v, f, vm, fm = load_mesh(input_file, load_uv=True, device=device)
    v = v + 1
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
    M = compute_matrix(v, f, lambda_smooth)

    position_145 = position[145]
    # position_224 = position[224]

    profiler_start()
    animation = [to_numpy(v)]
    pbar = tqdm(total=steps * substeps * opt_iter)
    for i in range(steps):
        tqdm.write(f'\nTimestep: {(i+1) * time:.3f}')
        for j in range(substeps):
            velocity = velocity + delta_t * gravity
            previous = position
            position = position + delta_t * velocity
            position[145] = position_145
            y = position[..., 1]
            position[..., 1] = y * (y > 0)

            # this part can also be considered as an optimization problem
            # solve constraints: with conditional gradient descent for optimization purpose
            p = to_differential(M, position)
            p = p.requires_grad_()
            o = AdamUniform([p], lr=lr)
            for k in range(opt_iter):
                q = from_differential(M, p)  # NOTE: not possible to naively keep the vertices here?
                q[145] = position_145
                # q[224] = position_224

                y = q[..., 1]
                q[..., 1] = y * (y > 0)

                energy_stretch = stretch_energy(q[None], garment)  # fake batch dimension
                energy_bend = bending_energy(q[None], garment)

                l = energy_stretch + energy_bend

                o.zero_grad(set_to_none=True)
                l.backward()
                o.step()

                pbar.update(1)
                profiler_step()

            tqdm.write(f'Loss: {l.item()}')
            q = from_differential(M, p.detach())
            q[145] = position_145
            # position[224] = position_224
            y = q[..., 1]
            q[..., 1] = y * (y > 0)

            position = position * (1 - relaxation) + q * relaxation

            residual = position - previous
            velocity = residual / delta_t

        tqdm.write(f'Mean residual: {residual.norm(dim=-1).mean().item()}')
        tqdm.write(f'Max residual: {residual.norm(dim=-1).max().item()}')
        tqdm.write(f'Mean velocity: {velocity.norm(dim=-1).mean().item()}')
        tqdm.write(f'Max velocity: {velocity.norm(dim=-1).max().item()}')
        tqdm.write(f'Bending energy: {energy_bend.item() / material.bending_multiplier}')
        tqdm.write(f'Stretching energy: {energy_stretch.item() / material.stretch_multiplier}')

        animation.append(to_numpy(position))

    profiler_stop()
    animation = np.stack(animation)
    animated_mesh = dotdict()
    animated_mesh.animation = animation
    animated_mesh.faces = to_numpy(f)
    export_dotdict(animated_mesh, filename=output_file.format('dynamic').replace('.ply', '.npz'))

    # for i in range(len(animated_mesh.animation)):
    #     export_mesh(animated_mesh.animation[i], garment.f, filename=output_file.format(f'dynamic{i}').replace('.npz', '.ply'))


if __name__ == '__main__':
    profiling = False

    prof_cfg = dotdict()
    prof_cfg.enabled = profiling
    prof_cfg.record_dir = 'data/record/parallel_method'
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
