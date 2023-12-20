import itertools
from torch import nn
from typing import Iterator, Tuple, Mapping, Dict

from torch.optim import Adam, AdamW, SGD, LBFGS, Optimizer
from easyvolcap.engine import OPTIMIZERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *

OPTIMIZERS.register_module()(Adam)
OPTIMIZERS.register_module()(AdamW)
OPTIMIZERS.register_module()(SGD)
OPTIMIZERS.register_module()(LBFGS)


@OPTIMIZERS.register_module()
def ConfigurableOptimizer(named_params: Iterator[Tuple[str, nn.Parameter]],

                          # Default parameters
                          lr: float = 5e-3,
                          eps: float = 1e-15,
                          weight_decay: float = 0.0,

                          lr_table: dotdict = dotdict(),  # empty special learning rate table
                          eps_table: dotdict = dotdict(),  # empty table
                          weight_decay_table: dotdict = dotdict(),  # empty table

                          optimizer_cfg: dotdict = dotdict(type=Adam.__name__),
                          ) -> Optimizer:
    if isinstance(named_params, Iterator):
        first = next(named_params)
        if isinstance(first, Tuple):
            named_params = itertools.chain([first], named_params)
        elif isinstance(first, nn.Parameter):
            log(yellow(f'Passed in a list of parameters, assuming they are named sequentially.'))
            named_params = {str(i): first for i, first in enumerate(named_params)}.items()
        else:
            raise NotImplementedError
    elif isinstance(named_params, Dict):
        named_params = named_params.items()
    else:
        raise NotImplementedError

    lr_line = dotdict()
    lr_line.lr = lr
    lr_line.eps = eps
    lr_line.weight_decay = weight_decay
    if lr_line: log('Starting learning rate config:', line(lr_line))

    lr_line = dotdict()
    if len(lr_table): lr_line.lr = lr_table
    if len(eps_table): lr_line.eps = eps_table
    if len(weight_decay_table): lr_line.weight_decay = weight_decay_table
    if lr_line: log('Special learning rate config:', line(lr_line))

    params = []
    for key, value in named_params:
        if not value.requires_grad:
            continue  # skip non-optimizable paramters
        v_lr = lr
        v_eps = eps
        v_weight_decay = weight_decay
        keys = key.split('.')
        for item in keys:
            if item in lr_table:
                v_lr = lr_table[item]
                # log(f'Special lr {key}: {v_lr}')
                break
        for item in keys:
            if item in eps_table:
                v_eps = eps_table[item]
                # log(f'Special eps {key}: {v_eps}')
                break
        for item in keys:
            if item in weight_decay_table:
                v_weight_decay = weight_decay_table[item]
                # log(f'Special weight decay {key}: {v_weight_decay}')
                break
        params.append(
            dotdict(
                params=[value],
                lr=v_lr,
                v_eps=v_eps,
                weight_decay=v_weight_decay,
                name=item
            )
        )

    if not len(params):
        log(red('optimizer got an empty parameter list, assume you\'re testing'))
        return None

    return OPTIMIZERS.build(optimizer_cfg, params=params)
