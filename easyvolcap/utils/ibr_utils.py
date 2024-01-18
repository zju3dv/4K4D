import torch
import numpy as np

from torch import nn
from functools import lru_cache
from torchvision.io import decode_jpeg

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.image_utils import pad_image

# How do we easily wraps all stuff in the lru?
# Initialize the lru requires some change to global variables, better expose APIs
# Input should only be latent_index and view_index, also use globals to manage other inputs
g_batch: dotdict = None
g_feat_regs = dotdict(g_feat_reg=None, g_bg_feat_reg=None)
g_src_inps = dotdict(src_inp=None, bg_src_inp=None)


def g_cached_input(latent_index: int, view_index: int, key: str = 'src_inps'):  # MARK: INPLACE
    global g_src_inps
    g_src_inp = g_src_inps.src_inp if key == 'src_inps' else g_src_inps.bg_src_inp
    if g_src_inp.ndim == 2:  # B, N or B, 3, H, W
        g_src_inp = decode_jpeg(g_src_inp[0].cpu(), device='cuda')[None].float() / 255  # decode -> 3, H, W, uint8(0,255) -> float32(0,1)
    return g_src_inp


def g_cached_feature(latent_index: int, view_index: int, key: str = 'fg'):
    g_feat_reg = g_feat_regs.g_bg_feat_reg if key == 'bg' else g_feat_regs.g_feat_reg
    g_src_inp = g_src_inps.src_inp
    return g_feat_reg(g_src_inp * 2 - 1)  # S, C, H, W


def prepare_caches(maxsize: int = 512):
    global g_cached_feature, g_cached_input
    g_cached_input = lru_cache(maxsize)(g_cached_input)
    g_cached_feature = lru_cache(maxsize)(g_cached_feature)


# NOTE: Everything starting with `get` is a cached function


def get_src_inps(batch: dotdict, key: str = 'src_inps'):  # MARK: INPLACE # !: BATCH
    global g_src_inps
    if key not in batch.meta:
        return compute_src_inps(batch, key)
    src_inps = batch.meta[key]

    inps = []
    t = batch.meta.t_inds[0]  # B, -> ()
    src_inds = batch.meta.src_inds[0]  # B, S -> S,
    for i, v in enumerate(src_inds):
        if key == 'src_inps': g_src_inps.src_inp = src_inps[i]  # jpeg bytes or tensors
        else: g_src_inps.bg_src_inp = src_inps[i]  # jpeg bytes or tensors
        inp = g_cached_input(t.item(), v.item(), key)  # B, 3, H, W
        inps.append(inp)
    batch[key] = inps  # list of B, 3, H, W
    return compute_src_inps(batch, key)


def get_src_feats(src_inps: torch.Tensor, feat_reg: nn.Module, batch: dotdict, key: str = 'fg'):  # !: BATCH
    global g_src_inps, g_feat_regs
    if key == 'fg': g_feat_regs.g_feat_reg = feat_reg
    else: g_feat_regs.g_bg_feat_reg = feat_reg
    src_inps = src_inps[0]  # !: BATCH

    Hc, Wc = src_inps.shape[-2:]  # cropped image size
    Hp, Wp = int(np.ceil(Hc / feat_reg.size_pad)) * feat_reg.size_pad, int(np.ceil(Wc / feat_reg.size_pad)) * feat_reg.size_pad  # Input and output should be same in size
    Hps = [int(Hp * s + 1e-5) for s in feat_reg.scales]
    Wps = [int(Wp * s + 1e-5) for s in feat_reg.scales]

    feats = []
    t = batch.meta.t_inds[0]  # B, -> ()
    src_inds = batch.meta.src_inds[0]  # B, S -> S,
    for i, v in enumerate(src_inds):
        g_src_inps.src_inp = src_inps[i]  # use full image for both foreground and background (layer enerf experience)
        fs = g_cached_feature(t.item(), v.item(), key)
        fs = [pad_image(f, size=(h, w)) for h, w, f in zip(Hps, Wps, fs)]  # resizing to current input
        feats.append(fs)
    feats = [torch.stack([f[i] for f in feats])[None] for i in range(len(feat_reg.scales))]  # HACK: too hacky...

    return feats


# Plain function for loading or computing the src_inps and src_feats
# NOTE: Everything starting with `compute` is a plain function without cache

def compute_src_inps(batch: dotdict, key: str = 'src_inps'):
    # [Storage] Input image might have completely different size from each other from the get go
    # [Dataset] Given differently sized images, performs cropping and scaling, update intrinsics, store as jpeg bytes -> jpeg streams are small
    # [Sampler] Decodes stored images (using torchvision api of nvjpeg), stack them together with zero padding -> images might be large
    # [Sampler] Forwards images to interpolation, pass through feature decoder -> feature might be large

    # The `key` may be `src_inps` or `src_bkgs` for enerf and layer enerf respectively
    if key not in batch and key in batch.meta and isinstance(batch.meta[key], list) and batch.meta[key][0].ndim == 2:
        # Perform decoding
        # The input tensor must be on CPU when decoding with nvjpeg
        # decode_jpeg already outputs 3, H, W images, only accepts 1d uint8
        batch[key] = [torch.cat([decode_jpeg(inp.cpu(), device='cuda') for inp in inps])[None].float() / 255 for inps in batch.meta[key]]  # decode -> 3, H, W, uint8(0,255) -> float32(0,1)

    if isinstance(batch[key], list) and batch[key][0].ndim == 4:  # S: B, C, H, W -> B, S, C, H, W
        # Perform reshaping
        max_h = max([i.shape[-2] for i in batch[key]])
        max_w = max([i.shape[-1] for i in batch[key]])
        batch[key] = torch.stack([pad_image(img, [max_h, max_w]) for img in batch[key]]).permute(1, 0, 2, 3, 4)  # S: B, C, H, W -> B, S, C, H, W

    return batch[key].contiguous()  # always return a contiguous tensor


def compute_src_feats(src_inps: torch.Tensor, feat_reg: nn.Module, batch: dotdict = None):
    # [Sampler] Maybe perform convolution on the input src_inps (if training)
    # [Sampler] Or just load it from the dataset, is preloading feature the best practice?
    sh = src_inps.shape
    src_inps = src_inps.view(-1, *sh[-3:])
    feats = feat_reg(src_inps * 2 - 1)  # always return a tensor
    feats = [f.view(*sh[:-3], *f.shape[-3:]) for f in feats]
    return feats
