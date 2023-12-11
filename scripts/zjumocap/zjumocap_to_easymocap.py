import os
import argparse
import numpy as np

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.easy_utils import read_camera, write_camera
# fmt: on


def read_annots(annots_path):
    annots = np.load(annots_path, allow_pickle=True).item()
    cams = annots['cams']

    item = {}
    for i in range(0, len(cams['K'])):
        k = '{:02d}'.format(i)

        item[k] = {}
        # item[k]['R'] = np.linalg.inv(cams['R'][i])
        item[k]['R'] = cams['R'][i]
        # item[k]['T'] = -item[k]['R'] @ cams['T'][i] / 1000.0
        item[k]['T'] = cams['T'][i] / 1000.0
        item[k]['K'] = cams['K'][i]
        item[k]['dist'] = cams['D'][i]

    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/xuzhen/datasets/sou')
    args = parser.parse_args()
    data_dir = args.data_dir
    cameras = read_annots(os.path.join(data_dir, 'annots.npy'))
    write_camera(cameras, data_dir)


if __name__ == '__main__':
    main()
