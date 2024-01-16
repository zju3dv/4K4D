import math
import torch
from typing import Callable, Tuple
from easyvolcap.utils.console_utils import *

def chunkify(chunk_size=1024,
             key='ray_o',
             pos=0,
             dim=-2,
             merge_dims: bool = False,
             ignore_mismatch: bool = False,  # ignore mismatch in batch dims
             print_progress: bool = False,
             move_to_cpu: bool = False,
             batch_key: str = 'batch',
             inds_key: str = 'chunkify_sample',
             ):
    from easyvolcap.utils.data_utils import to_cpu, to_cuda, to_numpy, to_tensor  # keep global imports clean
    # will fail if dim == -1, currently only tested on dim == -2 or dim == 1
    # will select a key element from the argments: either by keyword `key` or position `pos`
    # then, depending on whether user wants to merge other dimensions, will select the dim to chunkify according to `dim`

    def merge_ret(ret, x: torch.Tensor, sh: torch.Size, nn_dim: int):
        # Merge ret list based on reture type (single tensor or dotdict?)
        # Return values of chunified function should all be tensors
        if len(ret) and isinstance(ret[0], torch.Tensor):
            # Stop recursion
            ret = torch.cat(ret, dim=nn_dim)
            if ignore_mismatch:
                ret = ret
            else:
                ret = ret.view(*sh, *ret.shape[nn_dim + 1:]) if x.shape[nn_dim] == ret.shape[nn_dim] else ret
        elif len(ret) and isinstance(ret[0], dict):
            dict_type = type(ret[0])
            # Start recursion
            ret = {k: merge_ret([v[k] for v in ret], x, sh, nn_dim) for k in ret[0].keys()}
            ret = dict_type(ret)
        elif len(ret) and (isinstance(ret[0], list) or isinstance(ret[0], tuple)):
            list_type = type(ret[0])
            # Start recursion
            ret = [merge_ret([v[i] for v in ret], x, sh, nn_dim) for i in range(len(ret[0]))]
            ret = list_type(ret)
        else:
            raise RuntimeError(f'Unsupported return type to batchify: {type(ret[0])}, or got empty return value')
        return ret

    def wrapper(decoder: Callable[[torch.Tensor], torch.Tensor]):
        def decode(*args, **kwargs):
            # Prepare pivot args (find shape information from this arg)
            if key in kwargs:
                x: torch.Tensor = kwargs[key]
            else:
                x: torch.Tensor = args[pos]
                args = [*args]
            sh = x.shape[:dim + 1]  # record original shape up until the chunkified dim
            nn_dim = len(sh) - 1  # make dim a non-negative number (i.e. -2 to 1?)

            # Prepare all tensor arguments by filtering with isinstance
            tensor_args = [v for v in args if isinstance(v, torch.Tensor)]
            tensor_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
            other_args = [v for v in args if not isinstance(v, torch.Tensor)]
            other_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}

            # Merge all dims except first batch dim up until the actual chunkify dimension
            if merge_dims:
                x = x.view(x.shape[0], -1, *x.shape[nn_dim + 1:])
                tensor_args = [v.view(v.shape[0], -1, *v.shape[nn_dim + 1:]) for v in tensor_args]
                tensor_kwargs = {k: v.view(v.shape[0], -1, *v.shape[nn_dim + 1:]) for k, v in tensor_kwargs.items()}
                nn_dim = 1  # will always be 1 in this situation

            # Running the actual batchified forward pass
            ret = []
            total_size = x.shape[nn_dim]
            # We need to update chunk size so that almost all chunk has a decent amount of queries
            actual_size = math.ceil(total_size / math.ceil(total_size / chunk_size)) if total_size else chunk_size  # this value should be smaller than the actual chunk_size specified
            if print_progress: pbar = tqdm(total=total_size, back=3)  # log previous frame
            for i in range(0, total_size, actual_size):
                # nn_dim should be used if there's multiplication involved
                chunk_args = [v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for v in tensor_args]
                chunk_kwargs = {k: v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for k, v in tensor_kwargs.items()}

                # Other components can use this to perform manual trunking
                if batch_key in other_kwargs: other_kwargs[batch_key].meta[inds_key] = [i, i + actual_size]

                result = decoder(*chunk_args, *other_args, **chunk_kwargs, **other_kwargs)
                result = to_cpu(result, non_blocking=True) if move_to_cpu else result
                ret.append(result)
                if print_progress: pbar.update(min(i + actual_size, total_size) - i)
            if print_progress: pbar.close()  # manual close necessary!

            if not len(ret):
                # Brute-forcely go through the network with empty input
                log(f'zero length tensor detected in chunkify, are the camera parameters correct?', 'red')
                i = 0
                chunk_args = [v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for v in tensor_args]
                chunk_kwargs = {k: v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for k, v in tensor_kwargs.items()}
                result = decoder(*chunk_args, *other_args, **chunk_kwargs, **other_kwargs)
                result = to_cpu(result, non_blocking=True) if move_to_cpu else result
                ret.append(result)

            return merge_ret(ret, x, sh, nn_dim)
        return decode
    return wrapper


def key_cache(key: Callable):
    def key_cache_wrapper(func: Callable):
        # will only use argument that match the key positiona or name in the args or kwargs collection as lru_cache's key
        cached_result = None
        cached_hash = None

        def func_wrapper(*args, **kwargs):
            nonlocal cached_result, cached_hash
            key_value = key(*args, **kwargs)
            key_hash = hash(key_value)
            if key_hash != cached_hash:
                cached_result = func(*args, **kwargs)
                cached_hash = key_hash
            return cached_result

        return func_wrapper
    return key_cache_wrapper


def batch_aware_indexing(mask: torch.Tensor, metric: torch.Tensor = None, dim=-1) -> Tuple[torch.Tensor, torch.Tensor, int]:  # MARK: SYNC
    # dim: in terms of the index (mask)
    if mask.dtype != torch.bool: mask = mask.bool()
    if metric is None: metric = mask.int()
    if metric.dtype == torch.bool: metric = metric.int()
    # retain all other dimensions (likely batch dimensions)
    S = mask.sum(dim=dim).max().item()  # the max value of this dim on all other dimension
    valid, inds = metric.topk(S, dim=dim, sorted=False)  # only find the top (mask = True) values (randomly select other values)
    return valid, inds, S


def multi_indexing(indices: torch.Tensor, shape: torch.Size, dim=-2):
    # index will first be augmented to match the values' dimentionality at the back
    # then we will try to broatcast index's shape to values shape
    shape = list(shape)
    back_pad = len(shape) - indices.ndim
    for _ in range(back_pad): indices = indices.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return indices.expand(*expand_shape)


def multi_gather(values: torch.Tensor, indices: torch.Tensor, dim=-2):
    # Gather the value at the -2th dim of values, augment index shape on the back
    # Example: values: B, P, 3, index: B, N, -> B, N, 3

    # index will first be augmented to match the values' dimentionality at the back
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the values's second to last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(indices, values.shape, dim))


def multi_scatter(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=-2):
    # backward of multi_gather
    return target.scatter(dim, multi_indexing(indices, values.shape, dim), values)


def multi_scatter_(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=-2):
    # inplace version of multi_scatter
    return target.scatter_(dim, multi_indexing(indices, values.shape, dim), values)


def multi_gather_tris(v: torch.Tensor, f: torch.Tensor, dim=-2) -> torch.Tensor:
    # compute faces normals w.r.t the vertices (considering batch dimension)
    if v.ndim == (f.ndim + 1): f = f[None].expand(v.shape[0], *f.shape)
    # assert verts.shape[0] == faces.shape[0]
    shape = torch.tensor(v.shape)
    remainder = shape.flip(0)[:(len(shape) - dim - 1) % len(shape)]
    return multi_gather(v, f.view(*f.shape[:-2], -1), dim=dim).view(*f.shape, *remainder)  # B, F, 3, 3


def linear_indexing(indices: torch.Tensor, shape: torch.Size, dim=0):
    assert indices.ndim == 1
    shape = list(shape)
    dim = dim if dim >= 0 else len(shape) + dim
    front_pad = dim
    back_pad = len(shape) - dim - 1
    for _ in range(front_pad): indices = indices.unsqueeze(0)
    for _ in range(back_pad): indices = indices.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return indices.expand(*expand_shape)


def linear_gather(values: torch.Tensor, indices: torch.Tensor, dim=0):
    # only taking linear indices as input
    return values.gather(dim, linear_indexing(indices, values.shape, dim))


def linear_scatter(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter(dim, linear_indexing(indices, values.shape, dim), values)


def linear_scatter_(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter_(dim, linear_indexing(indices, values.shape, dim), values)


def merge01(x: torch.Tensor):
    return x.reshape(-1, *x.shape[2:])


def scatter0(target: torch.Tensor, inds: torch.Tensor, value: torch.Tensor):
    return target.scatter(0, expand_at_the_back(target, inds), value)  # Surface, 3 -> B * S, 3


def gather0(target: torch.Tensor, inds: torch.Tensor):
    return target.gather(0, expand_at_the_back(target, inds))  # B * S, 3 -> Surface, 3


def expand_at_the_back(target: torch.Tensor, inds: torch.Tensor):
    for _ in range(target.ndim - 1):
        inds = inds.unsqueeze(-1)
    inds = inds.expand(-1, *target.shape[1:])
    return inds


def expand0(x: torch.Tensor, B: int):
    return x[None].expand(B, *x.shape)


def expand1(x: torch.Tensor, P: int):
    return x[:, None].expand(-1, P, *x.shape[1:])


def nonzero0(condition: torch.Tensor):
    # MARK: will cause gpu cpu sync
    # return those that are true in the provided tensor
    return condition.nonzero(as_tuple=True)[0]
