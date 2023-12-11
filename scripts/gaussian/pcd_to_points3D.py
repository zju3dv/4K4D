"""
This script will load and convert a .ply visual hull to a points3D file
"""

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.colmap_utils import write_points3D_text, Point3D
from easyvolcap.utils.data_utils import load_pts

import argparse
from os.path import join


@catch_throw
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', default='data/renbody/0013_01')
    parser.add_argument('--vhulls_dir', default='vhulls')
    parser.add_argument('--pcd_file', default='000000.ply')
    parser.add_argument('--out_dir', default='ngp/points3D.txt')
    args = parser.parse_args()

    vhull = join(args.data_root, args.vhulls_dir, args.pcd_file)
    outdir = join(args.data_root, args.out_dir)

    v, c, n, s = load_pts(vhull)

    p3d = {i: Point3D(id=i, xyz=v[i], rgb=(c[i] * 255).astype(int), error=0, image_ids=[0], point2D_idxs=[0]) for i in range(len(v))}
    write_points3D_text(p3d, outdir)


if __name__ == '__main__':
    main()
