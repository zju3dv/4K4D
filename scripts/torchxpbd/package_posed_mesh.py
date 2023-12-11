import argparse
from pytorch3d.io import load_ply

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import export_dotdict
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default='data/xuzhen36/talk/registration/deformation/optim_smplh.ply')  # a full mesh with verts and faces, blend weights and stuff
    parser.add_argument('-o', '--output_file', default='data/xuzhen36/talk/registration/deformation/optim_smplh.npz')
    args = parser.parse_args()
    
    verts, faces = load_ply(args.input_file)
    ret = dotdict()
    ret.verts = verts
    ret.faces = faces.contiguous() # FIXME: why sometimes ply returns uncontiguous faces
    export_dotdict(ret, args.output_file)


if __name__ == '__main__':
    main()
