# Load trained model
# Load camera parameters

# This function will try to invoke evc programmatically
import sys
from os.path import join
from functools import partial
from easyvolcap.utils import console_utils
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import to_numpy, to_tensor, to_cuda, to_cpu, remove_batch, add_batch
from easyvolcap.utils.easy_utils import write_camera, read_camera


@catch_throw
def main():
    # fmt: off
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t','test'] + evv_args + ['val_dataloader_cfg.dataset_cfg.skip_loading_images=True', 'val_dataloader_cfg.sampler_cfg.view_sample=0,None,1', 'val_dataloader_cfg.dataset_cfg.view_sample=0,None,1', 'val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1']  # disable log and use custom logging mechanism
    log = partial(console_utils.log, file=sys.stdout)  # monkey patch the actual output used in this script
    print = partial(console_utils.print, file=sys.stdout)  # monkey patch the actual output used in this script

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', default='optimized', help='The directory to put the optimized cameras in')
    args = parser.parse_args(our_args)

    # Entry point first, other modules later to avoid strange import errors
    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.engine import cfg
    from easyvolcap.engine import SAMPLERS

    from easyvolcap.dataloaders.datasamplers import get_inds
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    from easyvolcap.models.cameras.optimizable_camera import OptimizableCamera
    from easyvolcap.dataloaders.volumetric_video_dataloader import VolumetricVideoDataloader
    from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
    # fmt: on

    runner: VolumetricVideoRunner = test(cfg, dry_run=True)
    epoch = runner.load_network()
    runner.model.eval()  # load the required stuff

    # TODO: Handle avg export and loading for monocular dataset
    camera: OptimizableCamera = runner.model.camera
    dataloader: VolumetricVideoDataloader = runner.val_dataloader
    dataset: VolumetricVideoDataset = runner.val_dataloader.dataset

    cameras = dotdict()
    inds = get_inds(dataset)

    for index in inds.ravel().numpy().tolist():
        batch = add_batch(dataset.get_metadata(index))
        cam = dataset.camera_names[batch.meta.camera_index.item()]
        output = camera.forward_cams(batch)
        output.meta.clear()  # remove content in this
        cameras[cam] = to_numpy(remove_batch(output))

        # TODO: Handle export of optimized intrinsics, this is gonna be quite tricky
        # Since we apply the distortion beforehand
        # And sometimes we crop the images before update
        cameras[cam].H = dataset.cameras[cam][0].H
        cameras[cam].W = dataset.cameras[cam][0].W
        cameras[cam].K = dataset.cameras[cam][0].K
        cameras[cam].D = dataset.cameras[cam][0].D
        cameras[cam].ccm = dataset.cameras[cam][0].ccm

    camera_output = join(dataset.data_root, args.prefix)
    write_camera(cameras, camera_output)
    log(yellow(f'Optimized camera parameters written to: {blue(camera_output)}'), file=sys.stdout)


if __name__ == '__main__':
    main()
