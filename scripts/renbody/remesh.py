import os
import sys
import argparse
import trimesh
import numpy as np
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.mesh_utils import average_edge_length

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
REMESH_DIR = os.path.join(ROOT_DIR, '..', "data/remesh")
sys.path.append(REMESH_DIR)
from pyremesh import remesh_botsch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--edge_scale', '-es', type=float, default=1.0)
    args = parser.parse_args()

    mesh = trimesh.load_mesh(args.input)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    h = average_edge_length(vertices, faces, 'np') * args.edge_scale
    log(blue((f'botsch remesh. target edge length: {h:.4f}')))
    v_new, f_new = remesh_botsch(vertices.astype(np.double), faces.astype(np.int32), 5, h, True)

    mesh = trimesh.Trimesh(vertices=v_new, faces=f_new)
    mesh.export(args.output)
    

if __name__ == "__main__":
    main()