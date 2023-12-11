import os
import cv2
import argparse
import collections
import numpy as np
import ujson as json

from tqdm import tqdm
from os.path import join
from termcolor import cprint
from multiprocessing.pool import ThreadPool

# fmt: off
from easyvolcap.utils.colmap_utils import write_images_text, write_cameras_text, rotmat2qvec, Camera, Image
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.console_utils import *
# fmt: on


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='data/NHR/sport_1_easymocap')
parser.add_argument('--output_dir', default='ngp')
parser.add_argument('--image_dir', default='images')
parser.add_argument('--mask_dir', default='mask')
parser.add_argument('--intri', default='intri.yml')
parser.add_argument('--extri', default='extri.yml')
parser.add_argument('--image', default='000000.jpg')
parser.add_argument('--ratio', default=1.0, type=float)
parser.add_argument('--camera_model', default='OPENCV', help='This will also control whether or not we are performing undistortion.')
parser.add_argument('--no_use_mask', action='store_false', dest='use_mask')
parser.add_argument('--crop_imgs', action='store_true', help='Used in combination with args.use_mask. If this is set to true, will always use PINHOLE and perform undist first.')
parser.add_argument('--mask_bkgd', type=float, default=-1.0, help='Used in combination with args.use_mask.')
args = parser.parse_args()

# Prepare actual directories
args.output_dir = join(args.data_root, args.output_dir)
args.image_dir = join(args.data_root, args.image_dir)
args.mask_dir = join(args.data_root, args.mask_dir)
args.intri = join(args.data_root, args.intri)
args.extri = join(args.data_root, args.extri)

if args.crop_imgs: args.camera_model = 'PINHOLE'

# Maybe match easymocap CoreView_313 dataformat
cams = read_camera(args.intri, args.extri)
basenames = list(cams.keys())
os.system(f'mkdir -p {args.output_dir}')
images = {}
cameras = {}
miscs = {}
eval_images = {}
ext = os.path.splitext(args.image)[-1]


def load_image_and_camera(cam_id: int, cam_name: str):
    # Load camera parameters from extri.yml and intri.yml
    print(f'processing easymocap data for camera: {cam_name}')
    cam_dict = cams[cam_name]
    K = cam_dict['K']
    R = cam_dict['R']
    T = cam_dict['T']
    D = cam_dict['D']  # NOTE: losing k3 parameter
    if D.shape[0] == 1:
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[0, 1], D[0, 2], D[0, 3], D[0, 4], 0, 0, 0
    else:
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[1, 0], D[2, 0], D[3, 0], D[4, 0], 0, 0, 0
    qvec = rotmat2qvec(R)
    tvec = T.T[0]

    # Prepare image path for image.txt loading and saving
    name = f"{cam_name}{ext}"
    img_path = f'{args.image_dir}/{cam_name}/{args.image}'
    msk_path = f'{args.mask_dir}/{cam_name}/{args.image}'  # NOTE: possible jpg mask
    tar_path = f'{args.output_dir}/{name}'.replace('.jpg', '.png')
    name = name.replace('.jpg', '.png')

    # Maybe load training image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Only apply ratio on training images
    ori_H, ori_W, _ = img.shape
    H, W = int(ori_H * args.ratio), int(ori_W * args.ratio)
    y_ratio, x_ratio = H / ori_H, W / ori_W
    fx, fy, cx, cy = fx * x_ratio, fy * y_ratio, cx * x_ratio, cy * y_ratio
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # MARK: ratio

    misc = dotdict()

    # Maybe apply mask to alpha channel
    if args.use_mask:

        # Load mask and merge with image
        if not os.path.exists(msk_path): msk_path = msk_path.replace('.jpg', '.png')
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)  # MARK: ratio
        msk = (msk > 128).astype(np.uint8) * 255
        if args.mask_bkgd >= 0: img[msk < 128] = args.mask_bkgd * 255
        img = np.dstack([img, msk[..., None]])

        if args.crop_imgs:
            img = cv2.undistort(img, K, D)

            # msk = img[..., 3] # some of the undistortion will make images unrecognizable

            x, y, w, h = cv2.boundingRect(msk)
            img = img[y:y + h, x:x + w]
            misc.x = x
            misc.y = y
            misc.w = w
            misc.h = h
            H, W = img.shape[:2]
            misc.W = W
            misc.H = H
            cx -= x
            cy -= y

    cv2.imwrite(tar_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    height = img.shape[0]
    width = img.shape[1]

    image = Image(
        id=cam_id,
        qvec=qvec,
        tvec=tvec,
        camera_id=cam_id,
        name=name,
        xys=[],
        point3D_ids=[]
    )

    params = [fx, fy, cx, cy, k1, k2, p1, p2]
    camera = Camera(
        id=cam_id,
        model=args.camera_model,
        width=width,
        height=height,
        params=params
    )

    return camera, image, misc


cam_ids = list(range(len(basenames)))
cam_names = basenames

results = parallel_execution(cam_ids, cam_names, action=load_image_and_camera, print_progress=False, sequential=False)
for cam_id, (camera, image, misc) in zip(cam_ids, results):
    image = Image(
        id=image.id,
        qvec=image.qvec,
        tvec=image.tvec,
        camera_id=image.camera_id,
        name=os.path.basename(image.name),
        xys=image.xys,
        point3D_ids=image.point3D_ids
    )
    cameras[cam_id] = camera
    images[cam_id] = image
    miscs[cam_id] = misc

if len(cameras):
    write_cameras_text(cameras, join(args.output_dir, 'cameras.txt'))
    write_images_text(images, join(args.output_dir, 'images.txt'))
    with open(join(args.output_dir, 'miscs.json'), 'w') as f:
        json.dump(miscs, f)
