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
from easyvolcap.utils.net_utils import point_padding, affine_padding, affine_inverse, multi_gather, multi_scatter

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
    result_dir = args.result_dir
    occ_thresh = args.occ_thresh

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
        pts, rgb, prd, occ, dpt, dir = [], [], [], [], [], []
        # near, far = [], []
        for v in range(nv):
            batch = dataset[inds[v, f]]  # get the batch data for this view
            # Handle data movement
            batch = add_batch(to_cuda(batch))
            meta = batch.meta
            del batch.meta
            # batch = to_x(batch, runner.model.dtype)
            batch.meta = meta
            with torch.inference_mode() and torch.no_grad():
                output = runner.model(batch)  # get everything we need from the model, this performs the actual rendering

            rgb.append(batch.rgb.detach().cpu())
            prd.append(output.rgb_map.detach().cpu())
            occ.append(output.acc_map.detach().cpu())
            dpt.append(output.dpt_map.detach().cpu())
            pts.append((batch.ray_o + output.dpt_map * batch.ray_d).detach().cpu())
            dir.append(batch.ray_d.detach().cpu())

            # near.append(batch.near.detach().cpu())
            # far.append(batch.far.detach().cpu())

            pbar.update()

        prd = torch.cat(prd, dim=-2)
        pts = torch.cat(pts, dim=-2)
        rgb = torch.cat(rgb, dim=-2)
        occ = torch.cat(occ, dim=-2)
        dpt = torch.cat(dpt, dim=-2)
        dir = torch.cat(dir, dim=-2)

        # near = torch.cat(near, dim=-2)
        # far = torch.cat(far, dim=-2)

        log(f'Removing low density points')
        # ind = ((occ > occ_thresh) & (dpt > near) & (dpt < far))[0, ..., 0].nonzero()[..., 0]  # P,
        ind = (occ > occ_thresh)[0, ..., 0].nonzero()[..., 0]  # P,
        prd = multi_gather(prd, ind[None, ..., None]).float()  # B, P, C
        pts = multi_gather(pts, ind[None, ..., None]).float()  # B, P, C
        rgb = multi_gather(rgb, ind[None, ..., None]).float()  # B, P, C
        occ = multi_gather(occ, ind[None, ..., None]).float()  # B, P, C
        dpt = multi_gather(dpt, ind[None, ..., None]).float()  # B, P, C
        dir = multi_gather(dir, ind[None, ..., None]).float()  # B, P, C

        # Remove statistic outliers (il_ind -> inlier indices)
        log(f'Removing outliers')
        ind = remove_outlier(pts, K=50, std_ratio=4.0, return_inds=True)[0]  # P,
        prd = multi_gather(prd, ind[None, ..., None]).float()  # B, P, C
        pts = multi_gather(pts, ind[None, ..., None]).float()  # B, P, C
        rgb = multi_gather(rgb, ind[None, ..., None]).float()  # B, P, C
        occ = multi_gather(occ, ind[None, ..., None]).float()  # B, P, C
        dpt = multi_gather(dpt, ind[None, ..., None]).float()  # B, P, C
        dir = multi_gather(dir, ind[None, ..., None]).float()  # B, P, C

        filename = join(result_dir, runner.exp_name, runner.visualizer.save_tag, 'POINT', f'{prefix}{f:04d}.ply')

        if dataset.use_aligned_cameras and not args.skip_align:  # match the visual hull implementation
            mat = affine_padding(dataset.c2w_avg)  # 4, 4
            pts = (point_padding(pts) @ mat.mT)[..., :3]  # homo
        export_pts(pts, rgb, filename=filename)
        log(yellow(f'Fused points saved to {blue(filename)}, totally {cyan(pts.numel() // 3)} points'))
    pbar.close()


if __name__ == '__main__':
    main()
