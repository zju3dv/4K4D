# Given input point cloud locations, sample rgbras

# This function will try to invoke evc programmatically
import torch
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import to_numpy, to_tensor, export_pts, load_pts
from easyvolcap.utils.fcds_utils import update_points_features, update_features


@catch_throw
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_pcd', type=str, default='data/geometry/tcr4dv_sport1/SPLATS/frame0000.ply')
    parser.add_argument('--target_pcd', type=str, default='data/NHR/sport_1_easymocap/vhulls_backup/000000.ply')
    parser.add_argument('--output_pcd', type=str, default='data/geometry/tcr4dv_sport1_subsample/SPLATS/frame0000.ply')
    args = parser.parse_args()

    verts, colors, _, scalars = load_pts(args.source_pcd)  # we only care about the point locations
    verts, colors, scalars = to_tensor([verts, colors, scalars])
    feats = torch.cat([colors, *[v for v in scalars.values()]], dim=-1) # assume sorted

    tar, _, _, _ = load_pts(args.target_pcd)  # we only care about the point locations
    tar = torch.as_tensor(tar)

    feat = update_features(tar[None], verts[None], feats[None], radius=1.0, K=1)[0]
    col, feat = feat[..., :3], feat[..., 3:]
    dot = dotdict()
    for k, v in scalars.items():
        dot[k], feat = feat[..., :v.shape[-1]], feat[..., v.shape[-1]:]

    tar, col, dot = to_numpy([tar, col, dot])
    export_pts(pts=tar, color=col, scalars=dot, filename=args.output_pcd)

if __name__ == '__main__':
    main()
