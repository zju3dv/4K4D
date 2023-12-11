import os
import argparse
from os.path import join
from easymocap.mytools.camera_utils import read_camera

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.colmap_utils import write_cameras_binary, write_images_binary, Camera, Image, rotmat2qvec, qvec2rotmat, write_cameras_text, write_images_text

parser = argparse.ArgumentParser()
parser.add_argument('--data_root',        default='data/my_zjumocap/my_313/bkgd')
parser.add_argument('--output_dir',       default='text')
parser.add_argument('-i', '--intri',      default='intri.yml')
parser.add_argument('-e', '--extri',      default='extri.yml')
parser.add_argument('--height',           default=1024)
parser.add_argument('--width',            default=1024)
args = parser.parse_args()
args.output_dir = join(args.data_root, args.output_dir)
args.intri = join(args.data_root, args.intri)
args.extri = join(args.data_root, args.extri)
# fmt: on

cams = read_camera(args.intri, args.extri)
assert 'basenames' in cams and len(cams['basenames'])
basenames = sorted(cams['basenames'])
os.makedirs(args.output_dir, exist_ok=True)

cameras = {}
images = {}
for cam_id, cam_name in enumerate(basenames):
    print(f'reading image and camera from: {cam_name}')
    cam_dict = cams[cam_name]
    K = cam_dict['K']
    R = cam_dict['R']
    T = cam_dict['T']
    D = cam_dict['dist']  # !: losing k3 parameter
    if D.shape[0] == 1:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[0, 1], D[0, 2], D[0, 3], D[0, 4]
    else:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[1, 0], D[2, 0], D[3, 0], D[4, 0]

    params = [fx, fy, cx, cy, k1, k2, p1, p2]
    camera = Camera(
        id=cam_id,
        model='OPENCV',
        width=args.width,
        height=args.height,
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

write_cameras_text(cameras, join(args.output_dir, 'cameras.txt'))
write_images_text(images, join(args.output_dir, 'images.txt'))
