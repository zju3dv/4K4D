import torch
import numpy as np
import torch.nn.functional as F
from functools import lru_cache
from easyvolcap.utils.chunk_utils import linear_gather

# SCHP definitions
semantic_list = [
    'background',
    'hat',
    'hair',
    'glove',
    'sunglasses',
    'upper_cloth',
    'dress',
    'coat',
    'sock',
    'pant',
    'jumpsuit',
    'scarf',
    'skirt',
    'face',
    'left_leg',
    'right_leg',
    'left_arm',
    'right_arm',
    'left_shoe',
    'right_shoe',
]

semantic_dim = len(semantic_list)

# Conversion between the semantic map or the color used to represent those semantic maps


def color_to_semantic(schp: torch.Tensor,  # B, H, W, 3
                      palette: torch.Tensor,  # 256, 3
                      ):
    sem_msk = schp.new_zeros(schp.shape[:3])
    for i, rgb in enumerate(palette):
        belong = (schp - rgb).sum(axis=-1) == 0
        sem_msk[belong] = i
    return sem_msk  # B, H, W


def semantics_to_color(semantic: torch.Tensor,  # V,
                       palette: torch.Tensor,  # 256, 3
                       ):
    return linear_gather(palette, semantic)  # V, 3


def palette_to_index(sem: np.ndarray, semantic_dim=semantic_dim):
    # convert color coded semantic map to semantic index
    palette = get_schp_palette(semantic_dim)
    sem_msk = np.zeros(sem.shape[:2], dtype=np.uint8)
    for i, rgb in enumerate(palette):
        belong = (sem - rgb).sum(axis=-1) == 0
        sem_msk[belong] = i

    return sem_msk


def palette_to_onehot(sem: np.ndarray, semantic_dim=semantic_dim):
    sem_msk = palette_to_index(sem, semantic_dim)
    # convert semantic index to one-hot vectors
    sem = torch.from_numpy(sem_msk)
    sem: torch.Tensor = F.one_hot(sem.long(), semantic_dim)
    sem = sem.float().numpy()
    return sem


@lru_cache
def get_schp_palette(num_cls=256):
    # Copied from SCHP
    """ Returns the color map for visualizing the segmentation mask.
    Inputs:
        =num_cls=
            Number of classes.
    Returns:
        The color map.
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    palette = np.array(palette, dtype=np.uint8)
    palette = palette.reshape(-1, 3)  # n_cls, 3
    return palette

@lru_cache
def get_schp_palette_tensor_float(num_cls=semantic_dim, device='cuda'):
    # Copied from SCHP
    """ Returns the color map for visualizing the segmentation mask.
    Inputs:
        =num_cls=
            Number of classes.
    Returns:
        The color map.
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    palette = torch.tensor(palette, dtype=torch.float, device=device) / 255.0
    palette = palette.reshape(-1, 3)  # n_cls, 3
    return palette
