import numpy as np
import argparse, os, json
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nerfies_root', type=str, default='data/nerfies/toby-sit')
    parser.add_argument('--easyvolcap_root', type=str, default='data/nerfies/toby-sit')
    parser.add_argument('--ratio', type=int, default=2)
    args = parser.parse_args()


    # fetch basic information
    cam_prefix = ['left1', 'right1']
    cam_number = len(cam_prefix)  # 2


    # NOTE: important, process the frame number first
    # there will be some frames that only have one camera parameters
    frame_appear = os.listdir(join(args.nerfies_root, 'camera'))
    frame_appear = [x.split('.')[0] for x in frame_appear]
    frame_valids = []
    for frame_name in frame_appear:
        if cam_prefix[0] not in frame_name: continue
        if cam_prefix[1] + '_' + frame_name.split('_')[1] not in frame_appear: continue
        frame_valids.append(frame_name.split('_')[1])
    frame_valids = sorted(frame_valids)  # sorted frame ids


    # NOTE: important, load the global scene infomation first
    scene_info = json.load(open(join(args.nerfies_root, 'scene.json')))
    scale_factor = 1. / args.ratio
    scene_center = np.array(scene_info['center']).reshape(-1, 1)
    scene_scales = scene_info['scale']


    # transform and store the camera parameters as easymocap format 
    for cam_id, prefix in enumerate(cam_prefix):
        # store `intri.yml` and `extri.yml` for each frame
        mocap_camera = {}
        mocap_cam_path = join(args.easyvolcap_root, 'cameras', f'{cam_id:02d}')  # use 00, 01, 02... 10, otherwise the order will be wrong
        for frame_id, frame_name in enumerate(frame_valids):
            # path generation
            nerfi_cam_path = join(args.nerfies_root, 'camera', f'{prefix}_{frame_name}.json')
            os.makedirs(mocap_cam_path, exist_ok=True)

            # load the original camera parameters
            nerfi_camera = json.load(open(nerfi_cam_path, 'r'))
            mocap_camera[f'{frame_id:06d}'] = {
                'R': np.array(nerfi_camera['orientation']).astype(np.float32),
                'T': -np.array(nerfi_camera['orientation']) @ (((np.array(nerfi_camera['position']).reshape(-1, 1)) - scene_center) * scene_scales).astype(np.float32),
                'D': np.array([
                    nerfi_camera['radial_distortion'][0],
                    nerfi_camera['radial_distortion'][1],
                    nerfi_camera['tangential_distortion'][0],
                    nerfi_camera['tangential_distortion'][1],
                    nerfi_camera['radial_distortion'][2]
                ]).astype(np.float32).reshape(-1, 1),
                # 'W': nerfi_camera['image_size'][0] // args.ratio,
                # 'H': nerfi_camera['image_size'][1] // args.ratio,
                'K': np.array([
                    [nerfi_camera['focal_length'], 0, nerfi_camera['principal_point'][0]],
                    [0, nerfi_camera['focal_length'], nerfi_camera['principal_point'][1]],
                    [0, 0, args.ratio]
                ]).astype(np.float32) * scale_factor
            }
            if cam_id == 0 and frame_id == 0: print(mocap_camera)

    # write the easymocap format camera parameters
    write_camera(mocap_camera, mocap_cam_path)
    log(yellow(f'Converted cameras saved to {blue(join(mocap_cam_path, "{intri.yml,extri.yml}"))}'))
    print('Camera processing finished.')

    # craete the soft link for all the images to match easymocap format
    for cam_id, prefix in enumerate(cam_prefix):
        # create soft link for each image
        for frame_id, frame_name in enumerate(frame_valids):
            # TODO: make this run in parallel
            nerfi_img_path = join(args.nerfies_root, 'rgb', f'{args.ratio}x', f'{prefix}_{frame_name}.png')
            mocap_img_root = join(args.easyvolcap_root, 'images', f'{cam_id:02d}')
            os.makedirs(mocap_img_root, exist_ok=True)
            mocap_img_path = join(mocap_img_root, f'{frame_id:06d}.png')
            # nerfi_img_path = os.path.relpath(nerfi_img_path, mocap_img_root) # find the relative path of nerfies to easyvolcap
            cmd = f'ln -s {nerfi_img_path} {mocap_img_path}'
            os.system(cmd)
    print('Image processing finished.')

    # recenter and rescale the provided background points
    # NOTE: this is not necessary, but you need to do it if you want to use the background loss
    # TODO: Convert these to plys for easier visualization
    nerfi_bkgd_pts_path = join(args.nerfies_root, 'points.npy')
    mocap_bkgd_pts_path = join(args.easyvolcap_root, 'points.npy')
    bkgd_pts = np.load(nerfi_bkgd_pts_path).astype(np.float32)
    bkgd_pts = (bkgd_pts - scene_center.transpose(1, 0)) * scene_scales
    np.save(mocap_bkgd_pts_path, bkgd_pts)
    print('Background points processing finished.')
