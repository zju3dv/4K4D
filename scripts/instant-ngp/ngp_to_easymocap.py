import json
import msgpack
import numpy as np
from os.path import join

import argparse
from easymocap.mytools.camera_utils import read_camera, write_camera

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default='data/zju_mocap/my_377/colmap/opt_cam.msgpack')
parser.add_argument('-t', '--transforms', default='data/zju_mocap/my_377/colmap/transforms.json')
parser.add_argument('-o', '--output_dir', default='data/zju_mocap/my_377/colmap')
parser.add_argument('--ori_cam_dir', default='data/zju_mocap/my_377')  # where the original intri.yml and extri.yml resides, only replacing extrinsic
parser.add_argument('--no_replace_intri', action='store_false', dest='replace_intri')  # where the original intri.yml and extri.yml resides, only replacing extrinsic
parser.add_argument('--generate_new_cam', action='store_true')  # where the original intri.yml and extri.yml resides, only replacing extrinsic
args = parser.parse_args()
# will first try to read from the original intri.yml and extri.yml
cams = read_camera(join(args.ori_cam_dir, 'intri.yml'), join(args.ori_cam_dir, 'extri.yml'))
assert 'basenames' in cams and len(cams['basenames'])
if cams['basenames'][0].startswith('Camera_B'):
    prefix = 'Camera_B'
else:
    prefix = ''
basenames = sorted(cams['basenames'], key=lambda x: int(x[len(prefix):]))
del cams['basenames']

data = msgpack.load(open(args.input_file))
tran = json.load(open(args.transforms))
padding = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
scale = data['snapshot']['nerf']['dataset']['scale']
offset = np.array(data['snapshot']['nerf']['dataset']['offset'])  # 3,
ori_up = np.array(tran['ori_up'])
totp = np.array(tran['totp'])
avglen = np.array(tran['avglen'])

w = tran['w']
h = tran['h']


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


for cam_id, cam_name in enumerate(basenames):
    del cams[cam_name]['Rvec']  # let easymocap recompute Rvec
    c2w = np.array(data['snapshot']['camera']['transforms'][cam_id]['start'])  # start == end
    c2w = c2w[[2, 0, 1], :]  # swap back from ngp to nerf
    c2w[:, 2] *= -1  # swap back from ngp to nerf
    c2w[:, 1] *= -1  # swap back from ngp to nerf
    c2w[:, 3] = (c2w[:, 3] - offset) / scale
    c2w = np.concatenate([c2w, padding], 0)  # 4, 4 # restore to transforms.json format

    c2w[0:3, 3] /= 4. / avglen
    c2w[0:3, 3] += totp
    R = rotmat(ori_up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    c2w = np.matmul(R.T, c2w)  # restore to c2w format

    c2w[2, :] *= -1  # flip whole world upside down
    c2w = c2w[[1, 0, 2, 3], :]  # swap back y and z:w
    c2w[0:3, 2] *= -1  # flip the y and z axis
    c2w[0:3, 1] *= -1

    w2c = np.linalg.inv(c2w)  # 4, 4
    R, T = w2c[:3, :3], w2c[:3, 3:]  # restore to easymocap format

    print(f'{cam_name}:')
    print(f'R:\n{R}')
    print(f'original R:\n{cams[cam_name]["R"]}')
    print(f'T:\n{T}')
    print(f'original T:\n{cams[cam_name]["T"]}')
    cams[cam_name]['R'] = R
    cams[cam_name]['T'] = T

    if args.replace_intri:
        fx, fy = data['snapshot']['camera']['metadata'][cam_id]['focal_length']
        cx, cy = data['snapshot']['camera']['metadata'][cam_id]['principal_point']
        cx, cy = cx * w, cy * h
        k1, k2, p1, p2 = data['snapshot']['camera']['metadata'][cam_id]['camera_distortion'].values()

        K = cams[cam_name]['K'].copy()  # by reference
        D = cams[cam_name]['dist'].copy()  # by reference
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[0, 1], D[0, 2], D[0, 3], D[0, 4], 0, 0, 0
        if D.shape[0] == 1:
            K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[0, 1], D[0, 2], D[0, 3] = fx, fy, cx, cy, k1, k2, p1, p2
        else:
            K[0, 0], K[1, 1], K[0, 2], K[1, 2], D[0, 0], D[1, 0], D[2, 0], D[3, 0] = fx, fy, cx, cy, k1, k2, p1, p2
        print(f'K:\n{K}')
        print(f'original K:\n{cams[cam_name]["K"]}')
        print(f'D:\n{D}')
        print(f'original D:\n{cams[cam_name]["dist"]}')
        cams[cam_name]['K'] = K
        cams[cam_name]['dist'] = D

    print()

write_camera(cams, args.output_dir)

if args.generate_new_cam:
    new_cams = {}
    for cam_id, cam_name in enumerate(basenames):
        new_name = f'{cam_id:02d}'
        new_cams[new_name] = cams[cam_name]
    write_camera(new_cams, join(args.output_dir, 'new_cams'))
