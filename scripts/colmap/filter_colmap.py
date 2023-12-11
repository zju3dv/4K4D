import os
import torch
import numpy as np

import argparse
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.colmap_utils import read_model, write_model, qvec2rotmat

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='/mnt/data_fly/sparse')
parser.add_argument('--output', default='/mnt/data_fly/zjuht/distorted/sparse/0')
parser.add_argument('--bounds', nargs=6, default=[-362, -1037, -1000, 687, 92, 252])  # the 3d bounding box
args = parser.parse_args()

log('Reading colmap reconstruction from', blue(args.input))
cameras, images, points3D = read_model(args.input)
# bounds = torch.as_tensor(args.bounds, device=args.device).view(2, 3)
bounds = np.asarray(args.bounds).reshape(2, 3)

# 999.195, 291.267, 16.8698
# -1050.83, 259.682, -50.1964
# -246.033, 92.6426, -102.203
# 528.236, -259.083, 279.0u7

# 333.497, -1101.3, 246.486
# 338.174, -1102.63, 246.499
# 305.001, -762.761, 740.678
# -225.813, -1076.78, 246.966
# -115.965, -218.397, 237.494
# 589.067, -424.213, 247.113

# cameras = [cameras[k] for k in sorted(cameras.keys())]
# points3D = [points3D[k] for k in sorted(points3D.keys())]

# cameras = {k:cameras[k] for k in tqdm(cameras) if cameras[k]}

# 403.742, -1336.27, 245.203
# 628.981, -313.073, 245.228
# -295.317, -48.8139, 249.701
# -507.764, -1222.55, 246.976

# -500, -1336, -100, 628, -48, 300 # xmin, ymin, zmin, xmax, ymax, zmax
# -100 is for the cameras
# 300 is for the points
# the z axis should not matter
# breakpoint()
images = {k: images[k] for k in tqdm(images)
          if (-qvec2rotmat(images[k].qvec).T @ images[k].tvec > bounds[0]).all() and
          (-qvec2rotmat(images[k].qvec).T @ images[k].tvec < bounds[1]).all()
          }
points3D = {k: points3D[k] for k in tqdm(points3D) if (points3D[k].xyz > bounds[0]).all() and (points3D[k].xyz < bounds[1]).all()}

run(f'mkdir -p {args.output}')

write_model(cameras, images, points3D, args.output)
