"""
This script will load and convert a .ply visual hull to a points3D file
"""

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_pts, export_pts

import argparse
from os.path import join
import numpy as np


@catch_throw
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', default='data/enerf_outdoor/actor2_3')
    parser.add_argument('--vhulls_dir', default='merged')
    parser.add_argument('--vhulls_dirs', default=['vhulls', 'bkgd/boost'])
    parser.add_argument('--pcd_file', default='000000.ply')
    args = parser.parse_args()

    vs = []
    out = join(args.data_root, args.vhulls_dir, args.pcd_file)
    for vhull_dir in args.vhulls_dirs:
        vhull = join(args.data_root, vhull_dir, args.pcd_file)
        v, c, n, s = load_pts(vhull)
        vs.append(v)
    vs = np.concatenate(vs, axis=0)
    export_pts(vs, filename=out)


if __name__ == '__main__':
    main()
