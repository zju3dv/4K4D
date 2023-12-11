# run pifuhd on image to generate the view normal
# git clone the repo from: https://github.com/facebookresearch/pifuhd
# run its model weight download scripts: https://github.com/facebookresearch/pifuhd/blob/main/scripts/download_trained_model.sh

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
import torch.nn.functional as F

# import hacks
# fmt: off
import sys
sys.path.append('3rdparty/pifuhd')
from lib.model import HGPIFuNetwNML, HGPIFuMRNet # type: ignore
sys.path.pop()
for key in list(sys.modules.keys()):
    if key.startswith('lib'):
        del sys.modules[key]
sys.path.append('.')
from easyvolcap.utils.console_utils import log
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_image, save_image, tensor_to_list, list_to_tensor, numpy_to_list, list_to_numpy, to_numpy, to_cuda, load_mask
sys.path.pop()
# fmt: on

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_root', type=str, default='data/my_zjumocap/my_313')
parser.add_argument('-c', '--chkpt_path', type=str, default='/nas/home/xuzhen/weights/pifuhd/pifuhd.pt')
parser.add_argument('-i', '--input_file', type=str, default='image_list.txt', help='path to the image or text file containing the images')
parser.add_argument('-m', '--input_mask', type=str, default='mask_list.txt', help='path to the image or text file containing the mask')
parser.add_argument('-o', '--output_dir', type=str, default='normal')
parser.add_argument('-p', '--prefix_dir', type=str, default='images')
parser.add_argument('-k', '--chunk_size', type=int, default=256)
parser.add_argument('-b', '--batch_size', type=int, default=8)
args = parser.parse_args()
args.input_file = join(args.data_root, args.input_file)
args.input_mask = join(args.data_root, args.input_mask)
args.output_dir = join(args.data_root, args.output_dir)
args.prefix_dir = join(args.data_root, args.prefix_dir)


def extract_paths(input_file):
    paths = []
    with open(input_file, 'r') as f:
        for line in f:
            path = line.strip()
            paths.append(path)
    return paths


# prepare input path
input_paths = extract_paths(args.input_file)
if os.path.exists(args.input_mask):
    mask_paths = extract_paths(args.input_mask)

# prepare output path
os.system(f'mkdir -p {args.output_dir}')
if args.prefix_dir in input_paths[0]:
    output_paths = [path.replace(args.prefix_dir, args.output_dir) for path in input_paths]
else:
    log(f'could not match prefix {args.prefix_dir} in path sample: {input_paths[0]}', 'red')
    log(f'will use os.path.basename: {os.path.basename(input_paths[0])}', 'red')

log('loading network')
state_dict = torch.load(args.chkpt_path, map_location='cuda')
opt = state_dict['opt']
opt_netG = state_dict['opt_netG']
netG = HGPIFuNetwNML(opt_netG, 'orthogonal').cuda()
netMR = HGPIFuMRNet(opt, netG, 'orthogonal').cuda()
netMR.load_state_dict(state_dict['model_state_dict'])
netG.eval()

log('running forward pass')
pbar = tqdm(total=len(input_paths))
for i in range(0, len(input_paths), args.chunk_size):
    # load images
    img = list_to_numpy(parallel_execution(input_paths[i:i + args.chunk_size], action=load_image))  # NOTE: assuming last channel to be masked
    # maybe load mask
    if mask_paths is not None and img.shape[-3] == 3:
        msk = list_to_numpy(parallel_execution(mask_paths[i:i + args.chunk_size], action=load_mask))  # NOTE: assuming last channel to be masked
        img = np.concatenate([img, msk], axis=-3)
    assert img.shape[-3] == 4

    nml = []
    for j in range(0, img.shape[0], args.batch_size):
        with torch.no_grad():
            # extract batch data
            bat = img[j:j + args.batch_size]
            bat = to_cuda(bat)

            # prepare batch data
            msk = bat[:, -1:] > 0.5
            bat = bat[:, :-1]
            bat = bat * msk

            # network forward pass
            # bat need to be divisible by 16
            H, W = bat.shape[-2:]
            bat = F.pad(bat, (0, (16 - W % 16) % 16, 0, (16 - H % 16) % 16), mode='reflect')
            res = netG.netF(bat)
            res = res[..., :H, :W]

            # post processing on batch data
            res = res * 0.5 + 0.5
            res = torch.cat([res, msk], dim=-3)
            nml.append(to_numpy(res))
        pbar.update(args.batch_size)
    nml: np.ndarray = np.concatenate(nml, axis=0)
    torch.cuda.synchronize()  # finish up data transfers before writing to disk (otherwise would all be blank)
    parallel_execution(output_paths[i:i + args.chunk_size], numpy_to_list(nml), action=save_image, quality=95)
