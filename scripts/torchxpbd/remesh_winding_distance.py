import torch
import argparse
import numpy as np

from pytorch3d.io import load_ply, load_obj

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.mesh_utils import hierarchical_winding_distance_remesh
from easyvolcap.utils.data_utils import export_dotdict, export_mesh, load_mesh
# fmt: on


def winding_distance_remesh_file(
    input_file: str = 'sphere_hole.ply',
    output_file: str = 'remesh.ply',
    device='cuda',
    **kwargs,
):

    # Robustly load the mesh from provided file (ply, obj or npz)
    verts, faces = load_mesh(input_file, device)
    # Hierarchical remeshing: concatenate multiple parts and fix self-intersection of the mesh
    verts, faces = hierarchical_winding_distance_remesh(verts, faces, **kwargs)
    # Robustly save remeshed file to (ply, obj or npz)
    export_mesh(verts, faces, filename=output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default='data/xuzhen36/talk/registration/deformation/semantic_mesh.npz')
    parser.add_argument('-o', '--output_file', default='remesh.ply')
    parser.add_argument('opts', default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()
    opts = {args.opts[i]: float(args.opts[i+1]) for i in range(0, len(args.opts), 2)} # naively considering all options to be float parameters

    winding_distance_remesh_file(args.input_file, args.output_file, **opts)


if __name__ == '__main__':
    main()
 