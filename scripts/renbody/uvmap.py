import os
import sys
import argparse
import trimesh
import xatlas
import numpy as np
from easyvolcap.utils.console_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()

    mesh = trimesh.load_mesh(args.input)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    log(blue((f'xatlas uv parameterizing')))
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    xatlas.export(args.output, vertices[vmapping], indices, uvs)

if __name__ == "__main__":
    main()