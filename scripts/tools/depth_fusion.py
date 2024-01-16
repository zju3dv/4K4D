"""
Load marigold depth
Load easyvolcap cameras
Produce a back-projected and merged point cloud
"""

import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.ray_utils import get_rays
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.image_utils import resize_image
from easyvolcap.utils.data_utils import to_cuda, to_tensor, to_x, add_batch, load_image, export_pts


def load_depth():
    pass


@catch_throw
def main():
    data_root = 'data/bytedance/round3'
    depth_dir = 'bkgd/depth/depth_npy'
    depth_postfix = '_pred.npy'  # camera + postfix = depth file name
    images_dir = 'bkgd/images'
    image = '000000.jpg'
    output = f'bkgd/depth/{image.replace(".jpg", ".ply")}'
    ratio = 0.10
    scale = 10.0
    locals().update(vars(build_parser(locals()).parse_args()))

    xyzs = []
    rgbs = []
    cameras = read_camera(data_root)
    for cam in cameras:
        depth_file = join(data_root, depth_dir, cam + depth_postfix)
        image_file = join(data_root, images_dir, cam, image)
        rgb = to_tensor(load_image(image_file))  # H, W, 3

        dpt = to_tensor(np.load(depth_file)[..., None])  # H, W, 1

        H, W = dpt.shape[:2]
        K, R, T = cameras[cam].K, cameras[cam].R, cameras[cam].T
        K, R, T = to_tensor([K, R, T])

        H, W = int(H * ratio), int(W * ratio)
        K[:2] *= ratio
        rgb = resize_image(rgb, size=(H, W))
        dpt = resize_image(dpt, size=(H, W))

        ray_o, ray_d = get_rays(H, W, K, R, T, z_depth=True)
        xyz = ray_o + dpt * ray_d * scale
        xyzs.append(xyz)
        rgbs.append(rgb)

    xyz = torch.cat(xyzs, dim=-2)
    rgb = torch.cat(rgbs, dim=-2)
    filename = join(data_root, output)
    export_pts(xyz, rgb, filename=filename)
    log(yellow(f'Merged depth point cloud saved to {blue(filename)}'))


if __name__ == '__main__':
    main()
