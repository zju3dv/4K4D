import os
import torch
import numpy as np

import argparse
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.colmap_utils import read_model, write_model

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', nargs='+', default=['/home/xuzhen/local/gaussian-splatting/data/not_moonlight/sparse/0', '/home/xuzhen/local/gaussian-splatting/data/moonlight/sparse/0'])
parser.add_argument('--output', default='/mnt/data_fly/sparse')
args = parser.parse_args()

log('Reading colmap reconstruction from', blue(args.inputs))
cameras, images, points3D = zip(*[read_model(i) for i in args.inputs])
cameras = {k, v for k, v in cameras.items()}
images = {k, v for k, v in images.items()}
points3D = {k, v for k, v in points3D.items()}

# cameras, images, points3D = read_model(args.input)
# # bounds = torch.as_tensor(args.bounds, device=args.device).view(2, 3)
# bounds = np.asarray(args.bounds).reshape(2, 3)
# 
# # 999.195, 291.267, 16.8698
# # -1050.83, 259.682, -50.1964
# # -246.033, 92.6426, -102.203
# # 528.236, -259.083, 279.0u7
# 
# # cameras = [cameras[k] for k in sorted(cameras.keys())]
# # points3D = [points3D[k] for k in sorted(points3D.keys())]
# 
# # cameras = {k:cameras[k] for k in tqdm(cameras) if cameras[k]}
# images = {k:images[k] for k in tqdm(images) if (images[k].tvec > bounds[0]).all() and (images[k].tvec < bounds[1]).all()}
# points3D = {k:points3D[k] for k in tqdm(points3D) if (points3D[k].xyz > bounds[0]).all() and (points3D[k].xyz < bounds[1]).all()}

write_model(cameras, images, points3D, args.output)
