from os.path import join
import torch
import json
from argparse import Namespace
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.gaussian_utils import GaussianModel


@catch_throw
def main():
    # fmt: off
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t', 'test'] + evv_args

    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.engine import cfg
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    # fmt: on

    runner: VolumetricVideoRunner = test(cfg, dry_run=True)
    runner.load_network()
    runner.model.eval()

    ep_iter = runner.ep_iter
    epochs = runner.epochs
    iters = ep_iter * epochs

    data_root = cfg.dataloader_cfg.dataset_cfg.data_root
    out_root = join(data_root, 'gaussians')
    os.makedirs(out_root, exist_ok=True)

    sampler = runner.model.sampler
    for i, pcd in enumerate(sampler.pcds):
        pcd: GaussianModel
        mesh_name = os.path.basename(sampler.meshes[i]).split('.')[0]
        out_dir = join(out_root, mesh_name)
        pcd.save_ply(join(out_dir, f'point_cloud/iteration_{iters}', 'point_cloud.ply'))
        cfg_args = Namespace(
            sh_degree=pcd.active_sh_degree.int().item(),
            white_background=False,
            model_path="",
            source_path="",
            data_device='cuda',
            images='images',
            resolution=1,
            eval=[0]
        )

        with open(join(out_dir, "cfg_args"), 'w') as f:
            f.write(str(cfg_args))

        dataset = runner.val_dataloader.dataset
        cameras = dataset.cameras
        cam_list = []
        for k, val in cameras.items():
            v = val[0]
            out = dotdict()
            out['id'] = int(k)
            out['img_name'] = str(k)
            out['width'] = int(v['W'])
            out['height'] = int(v['H'])
            out['position'] = (-v['R'].T @ v['T']).flatten().tolist()
            out['rotation'] = [x.tolist() for x in v['R'].T]
            out['fx'] = v['K'][0, 0]
            out['fy'] = v['K'][1, 1]
            cam_list.append(out)
        with open(join(out_dir, "cameras.json"), 'w') as f:
            json.dump(cam_list, f)

        # torch.save((pcd.capture(), iters), join(out_dir, f'{mesh_name}.pth'))
    
if __name__ == '__main__':
    main()
