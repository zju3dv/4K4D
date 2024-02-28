import os
import cv2

from easyvolcap.utils.console_utils import *
from easymocap.mytools.camera_utils import read_camera
from easyvolcap.utils.colmap_utils import write_cameras_binary, write_images_binary, Camera, Image, rotmat2qvec, qvec2rotmat, write_cameras_text, write_images_text


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/my_zjumocap/my_313/bkgd')
    parser.add_argument('--output_dir', default='colmap')
    parser.add_argument('--image_dir', default='images_calib')
    parser.add_argument('--image_ext', default='.jpg')
    parser.add_argument('--mask_dir', default='masks')
    parser.add_argument('--mask_ext', default='.jpg')
    parser.add_argument('-i', '--intri', default='intri.yml')
    parser.add_argument('-e', '--extri', default='extri.yml')
    args = parser.parse_args()
    args.output_dir = join(args.data_root, args.output_dir)
    args.intri = join(args.data_root, args.intri)
    args.extri = join(args.data_root, args.extri)

    cams = read_camera(args.intri, args.extri)
    assert 'basenames' in cams and len(cams['basenames'])
    basenames = sorted(cams['basenames'])
    os.makedirs(args.output_dir, exist_ok=True)

    cameras = {}
    images = {}
    sizes = {}
    frames = os.listdir(join(args.data_root, args.image_dir, basenames[0]))
    frames = sorted([x.split('.')[0] for x in frames])
    for frame in frames:
        output_dir = join(args.output_dir, frame)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(join(output_dir, 'images'), exist_ok=True)
        os.makedirs(join(output_dir, 'masks'), exist_ok=True)
        os.makedirs(join(output_dir, 'sparse'), exist_ok=True)
        for cam_id, cam_name in enumerate(basenames):
            cam_id = cam_id + 1
            # write images
            # if os.path.exists(join(output_dir, 'images', f'{cam_id:02d}{args.image_ext}')):
            os.remove(join(output_dir, 'images', f'{cam_id:02d}{args.image_ext}'))
            # if os.path.exists(join(output_dir, 'masks', f'{cam_id:02d}{args.mask_ext}')):
            os.remove(join(output_dir, 'masks', f'{cam_id:02d}{args.mask_ext}'))
            os.symlink(join('../../..', args.image_dir, cam_name, f'{frame}{args.image_ext}'), join(output_dir, 'images', f'{cam_id:02d}{args.image_ext}'))
            os.symlink(join('../../..', args.mask_dir, cam_name, f'{frame}{args.mask_ext}'), join(output_dir, 'masks', f'{cam_id:02d}{args.mask_ext}'))
            # read image
            if cam_name not in sizes.keys():
                img = cv2.imread(join(output_dir, 'images', f'{cam_id:02d}{args.image_ext}'))
                sizes[cam_name] = img.shape[:2]
            # read camera
            print(f'reading image and camera from: {cam_name}')
            cam_dict = cams[cam_name]
            K = cam_dict['K']
            R = cam_dict['R']
            T = cam_dict['T']
            D = cam_dict['dist']  # !: losing k3 parameter
            if 'H' in cam_dict.keys() or 'W' in cam_dict.keys():
                cam_dict['H'] = sizes[cam_name][0]
                cam_dict['W'] = sizes[cam_name][1]
            if D.shape[0] == 1:
                fx, fy, cx, cy, k1, k2, p1, p2, k3 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[0, 1], D[0, 2], D[0, 3], D[0, 4]
            else:
                fx, fy, cx, cy, k1, k2, p1, p2, k3 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[1, 0], D[2, 0], D[3, 0], D[4, 0]

            params = [fx, fy, cx, cy, k1, k2, p1, p2]
            camera = Camera(
                id=cam_id,
                model='OPENCV',
                width=cam_dict['W'],
                height=cam_dict['H'],
                params=params
            )

            qvec = rotmat2qvec(R)
            tvec = T.T[0]
            name = f"{cam_id:02d}.jpg"

            image = Image(
                id=cam_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=cam_id,
                name=name,
                xys=[],
                point3D_ids=[]
            )

            cameras[cam_id] = camera
            images[cam_id] = image

        write_cameras_text(cameras, join(output_dir, 'sparse', 'cameras.txt'))
        write_images_text(images, join(output_dir, 'sparse', 'images.txt'))
        with open(join(output_dir, 'sparse', 'points3D.txt'), 'w') as f:
            f.writelines(['# 3D point list with one line of data per point:\n'])


if __name__ == '__main__':
    main()
