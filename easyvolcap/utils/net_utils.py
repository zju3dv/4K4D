# Main
import os
import torch
import random
import numpy as np
import torch.nn.functional as F

from itertools import accumulate
from collections import defaultdict

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Typing
from types import MethodType
from typing import List, Callable, Tuple, Union, Dict

# Utils
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.math_utils import torch_inverse_3x3, affine_inverse, normalize

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer
    from easyvolcap.models.networks.multilevel_network import MultilevelNetwork


def print_shape(batch: dotdict):
    if isinstance(batch, dict):
        for k, v in batch.items():
            print(k)
            print_shape(v)
    elif isinstance(batch, list):
        for v in batch:
            print_shape(v)
    elif isinstance(batch, torch.Tensor):
        print(f'{batch.shape}')
    else:
        print(batch)


type_mapping = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.bool: np.bool_
}


def torch_dtype_to_numpy_dtype(torch_dtype):
    return type_mapping.get(torch_dtype, None)


def reduce_record_stats(record_stats: dotdict):
    reduced_stats = dotdict()
    for k, v in record_stats.items():
        if isinstance(v, torch.Tensor):
            reduced_stats[k] = v.item()  # MARK: will cause sync
        else:
            reduced_stats[k] = v
    return reduced_stats


def typed(input_to: torch.dtype = torch.float, output_to: torch.dtype = torch.float):
    from easyvolcap.utils.data_utils import to_x

    def wrapper(func: Callable):
        def inner(*args, **kwargs):
            args = to_x(args, input_to)
            kwargs = to_x(kwargs, input_to)
            ret = func(*args, **kwargs)
            ret = to_x(ret, output_to)
            return ret
        return inner
    return wrapper


class VolumetricVideoModule(nn.Module):
    # This module does not register 'network' as submodule
    def __init__(self, network: nn.Module = None, **kwargs) -> None:
        super().__init__()
        self.unregistered = [network]

        # Prepare fake forward sample function
        # Hacky forward function definition
        def sample(self, *args, **kwargs):
            if not len(kwargs): batch = args[-1]
            else: batch = kwargs.pop('batch', dotdict())
            self.forward(batch)
            return None, None, None, None
        if not hasattr(self, 'sample'): self.sample = MethodType(sample, self)
        if not hasattr(self, 'render'): self.render = MethodType(sample, self)
        if not hasattr(self, 'compute'): self.compute = MethodType(sample, self)

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        if hasattr(super(), 'render_imgui'):
            super().render_imgui(viewer, batch)

    @property
    def network(self):
        network: Union["VolumetricVideoNetwork", 'MultilevelNetwork'] = self.unregistered[0]
        return network


class GradientModule(nn.Module):
    # GradModule is a module that takes gradient based on whether we're in training mode or not
    # Avoiding the high memory cost of retaining graph of *not needed* back porpagation
    def __init__(self):
        super(GradientModule, self).__init__()

    def take_gradient(self, output: torch.Tensor, input: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False) -> torch.Tensor:
        return take_gradient(output, input, d_out, self.training or create_graph, self.training or retain_graph)

    def take_jacobian(self, output: torch.Tensor, input: torch.Tensor):
        with torch.enable_grad():
            outputs = output.split(1, dim=-1)
        grads = [self.take_gradient(o, input, retain_graph=(i < len(outputs))) for i, o in enumerate(outputs)]
        jac = torch.stack(grads, dim=-2)
        return jac


def get_function(f: Union[Callable, nn.Module, str]):
    if isinstance(f, str):
        try: return getattr(F, f)  # 'softplus'
        except AttributeError: pass
        try: return getattr(nn, f)()  # 'Identity'
        except AttributeError: pass
        # Using eval is dangerous, will never support that
    elif isinstance(f, nn.Module):
        return f  # nn.Identity()
    else:
        return f()  # nn.Identity


def modulize(f: Callable):
    return Modulized(f)


class Modulized(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor):
        return self.f(x)


def number_of_params(network: nn.Module):
    return sum([p.numel() for p in network.parameters() if p.requires_grad])


def make_params(params: torch.Tensor):
    return nn.Parameter(torch.as_tensor(params), requires_grad=True)


def make_buffer(params: torch.Tensor):
    return nn.Parameter(torch.as_tensor(params), requires_grad=False)


def take_jacobian(func: Callable, input: torch.Tensor, create_graph=False, vectorize=True, strategy='reverse-mode'):
    return torch.autograd.functional.jacobian(func, input, create_graph=create_graph, vectorize=vectorize, strategy=strategy)


def take_gradient(output: torch.Tensor,
                  input: torch.Tensor,
                  d_out: torch.Tensor = None,
                  create_graph: bool = True,
                  retain_graph: bool = True,
                  is_grads_batched: bool = False,
                  ):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(inputs=input,
                                outputs=output,
                                grad_outputs=d_output,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                only_inputs=True,
                                is_grads_batched=is_grads_batched,
                                )
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


class NoopModule(nn.Module):
    def __init__(self,):
        super().__init__()
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(f'{prefix}'):
                del state_dict[key]


class MLP(GradientModule):
    def __init__(self, input_ch=32, W=256, D=8, out_ch=257, skips=[4], actvn=nn.ReLU(), out_actvn=nn.Identity(),
                 init_weight=nn.Identity(), init_bias=nn.Identity(), init_out_weight=nn.Identity(), init_out_bias=nn.Identity(), dtype=torch.float):
        super(MLP, self).__init__()
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.skips = skips
        self.linears = []
        for i in range(D + 1):
            I, O = W, W
            if i == 0:
                I = input_ch
            if i in skips:
                I = input_ch + W
            if i == D:
                O = out_ch
            self.linears.append(nn.Linear(I, O, dtype=dtype))
        self.linears = nn.ModuleList(self.linears)
        self.actvn = get_function(actvn) if isinstance(actvn, str) else actvn
        self.out_actvn = get_function(out_actvn) if isinstance(out_actvn, str) else out_actvn

        for i, l in enumerate(self.linears):
            if i == len(self.linears) - 1: init_out_weight(l.weight.data)
            else: init_weight(l.weight.data)

        for i, l in enumerate(self.linears):
            if i == len(self.linears) - 1: init_out_bias(l.bias.data)
            else: init_bias(l.bias.data)

    def forward_with_previous(self, input: torch.Tensor):
        x = input
        for i, l in enumerate(self.linears):
            p = x  # store output of previous layer
            if i in self.skips:
                x = torch.cat([x, input], dim=-1)
            if i == len(self.linears) - 1:
                a = self.out_actvn
            else:
                a = self.actvn
            x = a(l(x))  # actual forward
        return x, p

    def forward(self, input: torch.Tensor):
        return self.forward_with_previous(input)[0]


def setup_deterministic(fix_random=True,  # all deterministic, same seed, no benchmarking
                        allow_tf32=False,  # use tf32 support if possible
                        deterministic=True,  # use deterministic algorithm for CNN
                        benchmark=False,
                        seed=0,  # only used when fix random is set to true
                        ):
    # https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16
    # https://huggingface.co/docs/transformers/v4.18.0/en/performance#tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # by default, tf32 support of CNNs is enabled
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    if fix_random:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)


def get_state_dict(state_dict: dotdict, prefix: str = ''):
    if len(prefix) and not prefix.endswith('.'): prefix = prefix + '.'
    d = dotdict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            d[k[len(prefix):]] = v
    return d


def load_pretrained(model_dir: str, resume: bool = True, epoch: int = -1, ext: str = '.npz', remove_if_not_resuming: bool = False, warn_if_not_exist: bool = False):
    if not resume:  # remove nothing here
        if remove_if_not_resuming:
            if isdir(model_dir) and len(os.listdir(model_dir)):  # only inform the use if there are files
                # log(red(f"Removing trained weights: {blue(model_dir)}"))
                try: run(f'rm -r {model_dir}')
                except: pass
        return None, None

    if not exists(model_dir):
        if warn_if_not_exist:
            log(red(f'Pretrained network: {blue(model_dir)} does not exist'))
        return None, None
    if isdir(model_dir):
        pts = [
            int(pt.split('.')[0]) for pt in os.listdir(model_dir) if pt != f'latest{ext}' and pt.endswith(ext) and pt.split('.')[0].isnumeric()
        ]
        if len(pts) == 0 and f'latest{ext}' not in os.listdir(model_dir):
            return None, None
        if epoch == -1:
            if f'latest{ext}' in os.listdir(model_dir):
                pt = 'latest'
            else:
                pt = max(pts)
        else:
            pt = epoch
        model_path = join(model_dir, f'{pt}{ext}')
    else:
        model_path = model_dir

    if ext == '.pt' or ext == '.pth':
        pretrained = dotdict(torch.load(model_path, 'cpu'))
    else:
        from easyvolcap.utils.data_utils import to_tensor
        pretrained = dotdict(model=to_tensor(dict(**np.load(model_path))), epoch=-1)  # the npz files do not contain training parameters

    return pretrained, model_path


def load_model(
    model: nn.Module,
    optimizer: Union[nn.Module, None] = None,
    scheduler: Union[nn.Module, None] = None,
    moderator: Union[nn.Module, None] = None,
    model_dir: str = '',
    resume: bool = True,
    epoch: int = -1,
    strict: bool = True,  # report errors when loading "model" instead of network
    skips: List[str] = [],
    only: List[str] = [],
    allow_mismatch: List[str] = [],
):

    pretrained, model_path = load_pretrained(model_dir, resume, epoch, '.pt',
                                             remove_if_not_resuming=True,
                                             warn_if_not_exist=False)
    if pretrained is None: return 0

    pretrained_model = pretrained['model']
    if skips:
        keys = list(pretrained_model.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_model[k]

    if only:
        keys = list(pretrained_model.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_model[k]

    for key in allow_mismatch:
        if key in model.state_dict() and key in pretrained_model:
            model_parent = model
            pretrained_parent = pretrained_model
            chain = key.split('.')
            for k in chain[:-1]:  # except last one
                model_parent = getattr(model_parent, k)
                pretrained_parent = pretrained_parent[k]
            last_name = chain[-1]
            setattr(model_parent, last_name, nn.Parameter(pretrained_parent[last_name], requires_grad=getattr(model_parent, last_name).requires_grad))  # just replace without copying

    (model if not isinstance(model, DDP) else model.module).load_state_dict(pretrained_model, strict=strict)
    if optimizer is not None and 'optimizer' in pretrained.keys(): optimizer.load_state_dict(pretrained['optimizer'])
    if scheduler is not None and 'scheduler' in pretrained.keys(): scheduler.load_state_dict(pretrained['scheduler'])
    if moderator is not None and 'moderator' in pretrained.keys(): moderator.load_state_dict(pretrained['moderator'])
    log(f'Loaded model {blue(model_path)} at epoch {blue(pretrained["epoch"])}')
    return pretrained['epoch'] + 1


def load_network(
    model: nn.Module,
    model_dir: str = '',
    resume: bool = True,  # when resume is False, will try as a fresh restart
    epoch: int = -1,
    strict: bool = True,  # report errors if something is wrong
    skips: List[str] = [],
    only: List[str] = [],
    prefix: str = '',  # will match and remove these prefix
    allow_mismatch: List[str] = [],
):
    pretrained, model_path = load_pretrained(model_dir, resume, epoch,
                                             remove_if_not_resuming=False,
                                             warn_if_not_exist=False)
    if pretrained is None:
        pretrained, model_path = load_pretrained(model_dir, resume, epoch, '.pth',
                                                 remove_if_not_resuming=False,
                                                 warn_if_not_exist=False)
    if pretrained is None:
        pretrained, model_path = load_pretrained(model_dir, resume, epoch, '.pt',
                                                 remove_if_not_resuming=False,
                                                 warn_if_not_exist=resume)
    if pretrained is None:
        return 0

    # log(f'Loading network: {blue(model_path)}')
    # ordered dict cannot be mutated while iterating
    # vanilla dict cannot change size while iterating
    pretrained_model = pretrained['model']

    if skips:
        keys = list(pretrained_model.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_model[k]

    if only:
        keys = list(pretrained_model.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_model[k]

    if prefix:
        keys = list(pretrained_model.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if k.startswith(prefix):
                pretrained_model[k[len(prefix):]] = pretrained_model[k]
            del pretrained_model[k]

    for key in allow_mismatch:
        if key in model.state_dict() and key in pretrained_model and not strict:
            model_parent = model
            pretrained_parent = pretrained_model
            chain = key.split('.')
            for k in chain[:-1]:  # except last one
                model_parent = getattr(model_parent, k)
                pretrained_parent = pretrained_parent[k]
            last_name = chain[-1]
            setattr(model_parent, last_name, nn.Parameter(pretrained_parent[last_name], requires_grad=getattr(model_parent, last_name).requires_grad))  # just replace without copying

    (model if not isinstance(model, DDP) else model.module).load_state_dict(pretrained_model, strict=strict)
    log(f'Loaded network {blue(model_path)} at epoch {blue(pretrained["epoch"])}')
    return pretrained["epoch"] + 1


def save_npz(model: nn.Module,
             model_dir: str = '',
             epoch: int = -1,
             latest: int = True,
             ):
    from easyvolcap.utils.data_utils import to_numpy
    npz_path = join(model_dir, 'latest.npz' if latest else f'{epoch}.npz')
    state_dict = model.state_dict() if not isinstance(model, DDP) else model.module.state_dict()
    param_dict = to_numpy(state_dict)  # a shallow dict
    os.makedirs(dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, **param_dict)
    log(yellow(f'Saved model {blue(npz_path)} at epoch {blue(epoch)}'))


def save_model(model: nn.Module,
               optimizer: Union[nn.Module, None] = None,
               scheduler: Union[nn.Module, None] = None,
               moderator: Union[nn.Module, None] = None,
               model_dir: str = '',
               epoch: int = -1,
               latest: int = False,
               save_lim: int = 5,
               ):

    model = {
        # Special handling for ddp modules (incorrect naming)
        'model': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
        'epoch': epoch
    }

    if optimizer is not None:
        model['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        model['scheduler'] = scheduler.state_dict()

    if moderator is not None:
        model['moderator'] = moderator.state_dict()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_path = join(model_dir, 'latest.pt' if latest else f'{epoch}.pt')
    torch.save(model, model_path)
    log(yellow(f'Saved model {blue(model_path)} at epoch {blue(epoch)}'))

    ext = '.pt'
    pts = [
        int(pt.split('.')[0]) for pt in os.listdir(model_dir) if pt != f'latest{ext}' and pt.endswith(ext) and pt.split('.')[0].isnumeric()
    ]
    if len(pts) <= save_lim:
        return
    else:
        removing = join(model_dir, f"{min(pts)}.pt")
        # log(red(f"Removing trained weights: {blue(removing)}"))
        os.remove(removing)


def root_of_any(k, l):
    for s in l:
        a = accumulate(k.split('.'), lambda x, y: x + '.' + y)
        for r in a:
            if s == r:
                return True
    return False


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def reset_optimizer_state(optimizer):
    optimizer.__setstate__({'state': defaultdict(dict)})


def update_optimizer_state(optimizer, optimizer_state):
    for k, v in optimizer_state.items():
        if v.new_params == None:
            continue
        val = optimizer.state[k].copy()
        exp_avg = torch.zeros_like(v.new_params)
        exp_avg[v.new_keep] = val['exp_avg'][v.old_keep]
        val['exp_avg'] = exp_avg
        exp_avg_sq = torch.zeros_like(v.new_params)
        exp_avg_sq[v.new_keep] = val['exp_avg_sq'][v.old_keep]
        val['exp_avg_sq'] = exp_avg_sq
        del optimizer.state[k]
        optimizer.state[v.new_params] = val


def get_max_mem():
    return torch.cuda.max_memory_allocated() / 2 ** 20
