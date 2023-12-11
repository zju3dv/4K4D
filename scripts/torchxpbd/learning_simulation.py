import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import SGD, Adam
from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix
from largesteps.parameterize import from_differential, to_differential
from rich import traceback
traceback.install()

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.engine import cfg
from easyvolcap.utils.console_utils import log
from easyvolcap.utils.net_utils import MLP, take_gradient
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.mesh_utils import loop_subdivision
from easyvolcap.utils.data_utils import export_dotdict, load_mesh, export_mesh, to_numpy, load_npz, export_dotdict, load_dotdict, to_cuda
from easyvolcap.utils.prof_utils import setup_profiler, profiler_start, profiler_step, profiler_stop
from easyvolcap.utils.physx_utils import Garment, StVKMaterial, gravity_energy, inertia_term_sequence, stretch_energy, bending_energy, inertia_term, dynamic_term
from easyvolcap.utils.loss_utils import l2
# fmt: on


def perform_simulation(
    time=1/60,  # 1s for the simulation (physics time)
    steps=360,  # 5s for the simulation (physics steps
    substeps=1,  # 1ms for simulation substep
    quality_step=1.0,

    # opt_iter * lr -> how compliant the mesh is
    relaxation=1.0,
    lr=4e-2,
    opt_iter=80,  # the number of iterations for optimization leads to compliance: i.e. stiffness of the simulation
    lambda_smooth=1,  # will make the object much silky or stiff
):
    global position, garment
    opt_iter = int(opt_iter * quality_step)
    lr = lr / quality_step
    delta_t = time / substeps

    velocity = torch.zeros_like(position)

    M = compute_matrix(v, f, lambda_smooth)
    position_145 = position[145]

    animation = []
    residuals = []
    velocities = []
    bending_gradients = []
    stretch_gradients = []

    pbar = tqdm(total=steps * substeps * opt_iter)
    for i in range(steps):
        log(f'\nTimestep: {(i+1) * time:.3f}')

        # compute input values
        initial = position.detach().requires_grad_()
        with torch.enable_grad():
            energy_stretch = stretch_energy(initial[None], garment)
            energy_bending = bending_energy(initial[None], garment)
        stretch_gradient = take_gradient(energy_stretch, initial)
        bending_gradient = take_gradient(energy_bending, initial)
        velocities.append(velocity)
        stretch_gradients.append(stretch_gradient.detach())
        bending_gradients.append(bending_gradient.detach())

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

                y = q[..., 1]
                q[..., 1] = y * (y > 0)

                energy_stretch = stretch_energy(q[None], garment)  # fake batch dimension
                energy_bend = bending_energy(q[None], garment)

                l = energy_stretch + energy_bend

                o.zero_grad(set_to_none=True)
                l.backward()
                o.step()

                pbar.update(1)

            log(f'Loss: {l.item()}')
            q = from_differential(M, p.detach())
            q[145] = position_145
            y = q[..., 1]
            q[..., 1] = y * (y > 0)

            position = position * (1 - relaxation) + q * relaxation

            residual = position - previous
            velocity = residual / delta_t

        log(f'Mean residual: {residual.norm(dim=-1).mean().item()}')
        log(f'Max residual: {residual.norm(dim=-1).max().item()}')
        log(f'Mean velocity: {velocity.norm(dim=-1).mean().item()}')
        log(f'Max velocity: {velocity.norm(dim=-1).max().item()}')
        log(f'Bending energy: {energy_bend.item() / material.bending_multiplier}')
        log(f'Stretching energy: {energy_stretch.item() / material.stretch_multiplier}')

        animation.append(position.detach())
        residuals.append(residual.detach())

    return torch.stack(animation), torch.stack(residuals), torch.stack(velocities), torch.stack(stretch_gradients), torch.stack(bending_gradients)


device = 'cuda'

input_file = 'tshirt.obj'
v, f, vm, fm = load_mesh(input_file, load_uv=True, device=device)
v = v + 1
material = StVKMaterial()
garment = Garment(v, f, vm, fm, material)
position = v
gravity = torch.tensor([0, -9.8, 0], device=device)

cache = 'learning_simulation.cache.npz'
keys = ["animation", "residuals", "velocities", "stretch_gradients", "bending_gradients"]
if os.path.exists(cache):
    dot = to_cuda(load_dotdict(cache), device)
else:
    ret = perform_simulation()
    dot = dotdict()
    for key, value in zip(keys, ret):
        dot[key] = value
    export_dotdict(dot, cache)

for key in keys:
    locals()[key] = dot[key]

animation: torch.Tensor
residuals: torch.Tensor
velocities: torch.Tensor
stretch_gradients: torch.Tensor
bending_gradients: torch.Tensor

input_ch = 3 + 3 + 3  # velocity + bending gradient + stretching gradient
out_ch = 3  # velocity for this timestep
translator = MLP(input_ch=input_ch, out_ch=out_ch, W=64, D=8).to(device, non_blocking=True)

network_path = 'translator.cache.pth'
residual_limit = 0.05
if os.path.exists(network_path):
    translator.load_state_dict(torch.load(network_path))
else:
    lr = 1e-3
    ep_iter = 100
    opt_iter = 1000
    opt = Adam(translator.parameters(), lr=lr)
    for i in tqdm(range(opt_iter)):
        input = torch.cat([velocities + 1/60 * gravity, stretch_gradients, bending_gradients], dim=-1)
        pred = translator(input).tanh() * residual_limit  # F, N, 9 -> F, N, 3
        loss = l2(pred, residuals) * len(input)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        log(f'Loss: {loss.item():.6g}', 'magenta')

    fitting = dotdict()
    fitting.animation = to_numpy(pred + torch.cat([v[None], animation[:-1]]))
    fitting.faces = to_numpy(f)
    export_dotdict(fitting, 'fitting.npz')

    torch.save(translator.state_dict(), network_path)


def perform_inference(
    time=1/60,  # 1s for the simulation (physics time)
    steps=360,  # 5s for the simulation (physics steps
    substeps=1,  # 1ms for simulation substep
):
    global position, garment, translator
    delta_t = time / substeps

    velocity = torch.zeros_like(position)

    animation = []

    pbar = tqdm(total=steps)
    for i in range(steps):
        log(f'\nTimestep: {(i+1) * time:.3f}')

        # compute input values
        initial = position.detach().requires_grad_()
        with torch.enable_grad():
            energy_stretch = stretch_energy(initial[None], garment)
            energy_bending = bending_energy(initial[None], garment)
        stretch_gradient = take_gradient(energy_stretch, initial)
        bending_gradient = take_gradient(energy_bending, initial)
        residual = translator(torch.cat([velocity + delta_t * gravity, stretch_gradient, bending_gradient], dim=-1)).tanh() * residual_limit
        residual[145] = 0  # fix another point
        pbar.update(1)
        velocity = residual / delta_t
        position = position + residual
        animation.append(position.detach())

        log(f'Mean residual: {residual.norm(dim=-1).mean().item()}')
        log(f'Max residual: {residual.norm(dim=-1).max().item()}')
        log(f'Mean velocity: {velocity.norm(dim=-1).mean().item()}')
        log(f'Max velocity: {velocity.norm(dim=-1).max().item()}')

    return torch.stack(animation)


position = v
with torch.no_grad():
    animation = perform_inference()

prediction = dotdict()
prediction.animation = to_numpy(torch.cat([v[None], animation[:-1]]))
prediction.faces = to_numpy(f)
export_dotdict(prediction, 'prediction.npz')
