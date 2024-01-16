import torch
from torch.nn import functional as F

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.chunk_utils import multi_gather
from easyvolcap.utils.fcds_utils import remove_outlier


def filter_global_points(points: dotdict[str, torch.Tensor] = dotdict()):

    def gather_from_inds(ind: torch.Tensor, scalars: dotdict()):
        return dotdict({k: multi_gather(v, ind[..., None]) for k, v in scalars.items()})

    # Remove NaNs in point positions
    ind = (~points.pts.isnan())[..., 0].nonzero()[..., 0]  # P,
    points = gather_from_inds(ind, points)

    # Remove low density points
    ind = (points.occ > 0.01)[..., 0].nonzero()[..., 0]  # P,
    points = gather_from_inds(ind, points)

    # Remove statistic outliers (il_ind -> inlier indices)
    ind = remove_outlier(points.pts[None], K=50, std_ratio=4.0, return_inds=True)[0]  # P,
    points = gather_from_inds(ind, points)

    return points
