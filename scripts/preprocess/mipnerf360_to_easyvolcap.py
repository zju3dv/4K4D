"""
Convert the mipnerf360 dataset to easyvolcap format
Read camera pose from poses_bounds.npy
Symlink image files into separated folders per camera
"""

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.data_utils import as_numpy_func
from easyvolcap.utils.math_utils import affine_inverse


@catch_throw
def main():
    args = dotdict(
        mipnerf360_root='data/mipnerf360',
        easyvolcap_root='data/mipnerf360',
        raw_images_dir='images',
        out_images_dir='images_easyvolcap',
        image_name_no_ext='000000',
        camera_pose='poses_bounds.npy',
        scenes=['bonsai', 'bicycle', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill'],
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    scenes = args.scenes
    for scene in tqdm(scenes):
        keys = []
        # for scene in scenes:
        for i, img in enumerate(sorted(os.listdir(join(args.mipnerf360_root, scene, args.raw_images_dir)))):
            # shutil.copy(join(args.mipnerf360_root, scene, args.raw_images_dir, img), join(args.easyvolcap_root, scene, args.out_images_dir, img))
            key = f'{i:06d}'
            src = join(args.mipnerf360_root, scene, args.raw_images_dir, img)
            tar = join(args.easyvolcap_root, scene, args.out_images_dir, key, args.image_name_no_ext + splitext(img)[-1]).lower()

            if exists(tar): os.remove(tar)
            os.makedirs(dirname(tar), exist_ok=True)
            os.symlink(relpath(src, dirname(tar)), tar)
            keys.append(key)

        # https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/datasets/llff.py#L177
        raw = np.load(join(args.mipnerf360_root, scene, args.camera_pose), allow_pickle=True)  # 21, 17
        poses = raw[:, :15].reshape(-1, 3, 5)  # N, 3, 5
        bounds = raw[:, -2:]  # N, 2
        # Step 1: rescale focal length according to training resolution
        H, W, F = poses[0, :, -1]  # original intrinsics, same for all images

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right down front"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], poses[..., :1], -poses[..., 2:3], poses[..., 3:4]], -1)  # (N_images, 3, 4) exclude H, W, focal
        cameras = dotdict()

        for i in range(len(poses)):
            key = keys[i]

            c2w = poses[i]
            w2c = as_numpy_func(affine_inverse)(c2w)

            cameras[key] = dotdict()
            cameras[key].R = w2c[:3, :3]
            cameras[key].T = w2c[:3, 3:]
            cameras[key].K = np.zeros_like(cameras[key].R)
            cameras[key].K[0, 0] = F
            cameras[key].K[1, 1] = F
            cameras[key].K[0, 2] = W / 2
            cameras[key].K[1, 2] = H / 2
            cameras[key].K[2, 2] = 1.0
            cameras[key].n = bounds[i, 0]  # camera has near and far
            cameras[key].f = bounds[i, 1]  # camera has near and far

        write_camera(cameras, join(args.easyvolcap_root, scene))
        log(yellow(f'Converted cameras saved to {blue(join(args.easyvolcap_root, scene, "{intri.yml,extri.yml}"))}'))


if __name__ == '__main__':
    main()
