"""
Load a easyvolcap model
Perform rendering on all images of a particular frame
Save the rendered rgb and depth value, along with maybe other attributes
Fuse the final rendered depth values as points into one
This function will try to invoke evc programmatically
"""
import torch
import argparse
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.fcds_utils import voxel_down_sample, remove_outlier
from easyvolcap.utils.data_utils import add_batch, to_cuda, export_pts, export_mesh, export_pcd, to_x
from easyvolcap.utils.math_utils import point_padding, affine_padding, affine_inverse
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter
from easyvolcap.utils.fusion_utils import filter_global_points

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner


@catch_throw
def main():
    # fmt: off
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t', 'test'] + evv_args + ['configs=configs/specs/vis.yaml,configs/specs/fp16.yaml', 'val_dataloader_cfg.dataset_cfg.skip_loading_images=False', 'model_cfg.apply_optcam=True']

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='data/geometry')
    parser.add_argument('--occ_thresh', type=float, default=0.01)
    parser.add_argument('--skip_align', action='store_true')
    parser.add_argument('--skip_density', action='store_true')
    parser.add_argument('--skip_outlier', action='store_true')
    parser.add_argument('--skip_near_far', action='store_true')
    parser.add_argument('--near_far_pad', type=float, default=0.0)
    args = parser.parse_args(our_args)

    # Entry point first, other modules later to avoid strange import errors
    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.engine import cfg
    from easyvolcap.engine import SAMPLERS
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    # fmt: on

    runner: VolumetricVideoRunner = test(cfg, dry_run=True)
    runner.load_network()
    runner.model.eval()
    fuse(runner, args)  # pcd is a dotdict with all attributes that we want to retain


def fuse(runner: "VolumetricVideoRunner", args: argparse.Namespace):
    from easyvolcap.dataloaders.datasamplers import get_inds

    dataset = runner.val_dataloader.dataset
    inds = get_inds(dataset)
    nv, nl = inds.shape[:2]
    prefix = 'frame'

    if dataset.closest_using_t:
        nv, nl = nl, nv
        prefix = 'view'
        inds = inds.transpose(0, 1)

    pbar = tqdm(total=nl * nv, desc=f'Fusing rendered RGBD')
    for f in range(nl):
        ptss, rgbs, prds, occs, dpts, dirs = [], [], [], [], [], []
        for v in range(nv):
            # Handle data movement
            batch = dataset[inds[v, f]]  # get the batch data for this view
            batch = add_batch(to_cuda(batch))

            # Running inference
            with torch.inference_mode(), torch.no_grad():
                output = runner.model(batch)  # get everything we need from the model, this performs the actual rendering

            # Get output point clouds
            pts = (batch.ray_o + output.dpt_map * batch.ray_d)[0]
            rgb = batch.rgb[0]
            prd = output.rgb_map[0]
            occ = output.acc_map[0]
            dpt = output.dpt_map[0]
            dir = batch.ray_d[0]

            # Filter local points

            # Store it into list
            prds.append(prd.detach().cpu()[0])  # reduce memory usage
            ptss.append(pts.detach().cpu()[0])  # reduce memory usage
            rgbs.append(rgb.detach().cpu()[0])  # reduce memory usage
            occs.append(occ.detach().cpu()[0])  # reduce memory usage
            dpts.append(dpt.detach().cpu()[0])  # reduce memory usage
            dirs.append(dir.detach().cpu()[0])  # reduce memory usage

            pbar.update()

        # Concatenate per-view depth map and other information
        prd = torch.cat(prds, dim=-2).float()
        pts = torch.cat(ptss, dim=-2).float()
        rgb = torch.cat(rgbs, dim=-2).float()
        occ = torch.cat(occs, dim=-2).float()
        dpt = torch.cat(dpts, dim=-2).float()
        dir = torch.cat(dirs, dim=-2).float()

        # Apply some global filtering
        points = filter_global_points(dotdict(prd=prd, pts=pts, rgb=rgb, occ=occ, dpt=dpt, dir=dir))
        log(f'Filtered {len(pts)} - {len(points.pts)} = {len(pts) - len(points.pts)} points globally')
        prd, pts, rgb, occ, dpt, dir = points.prd, points.pts, points.rgb, points.occ, points.dpt, points.dir

        # Align point cloud with the average camera, which is processed in memory, to make sure the stored files are consistent
        if dataset.use_aligned_cameras and not args.skip_align:  # match the visual hull implementation
            pts = (point_padding(pts) @ affine_padding(dataset.c2w_avg).mT)[..., :3]  # homo

        # Save final fused point cloud back onto the disk
        filename = join(args.result_dir, runner.exp_name, runner.visualizer.save_tag, 'POINT', f'{prefix}{f:04d}.ply')
        export_pts(pts, rgb, filename=filename)
        log(yellow(f'Fused points saved to {blue(filename)}, totally {cyan(pts.numel() // 3)} points'))
    pbar.close()


if __name__ == '__main__':
    main()
