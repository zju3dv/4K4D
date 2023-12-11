import argparse
import numpy as np
from smplx.lbs import batch_rigid_transform, batch_rodrigues


# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import to_cuda, export_dotdict
from easyvolcap.utils.blend_utils import forward_deform_lbs
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default='data/xuzhen36/talk/registration/deformation/optim_tpose.npz')  # a full mesh with verts and faces, blend weights and stuff
    parser.add_argument('-o', '--output_file', default='data/xuzhen36/talk/registration/deformation/optim_mesh.npz')
    parser.add_argument('--pose_index', type=int, default=0)
    args = parser.parse_args()

    # Load mesh data: vertices related and some other LBS related parameters
    tpose = np.load(args.input_file)
    tpose = dotdict(**tpose)
    tpose = to_cuda(tpose)
    verts = tpose.verts[None]

    deform = (tpose.largesteps_verts[args.pose_index] - tpose.verts)[None]

    weights = tpose.weights[None]
    parents = tpose.parents
    tjoints = tpose.tjoints[None]
    poses = tpose.poses[0][None]
    Rh = tpose.Rh[0][None]
    Th = tpose.Th[0][None]

    # Perform deformation and LBS (optimized by us earlier), since SCHP live in world coordinate system
    R = batch_rodrigues(Rh)
    T = Th[..., None].mT
    Rs = batch_rodrigues(poses.reshape(-1, 3)).view(poses.shape[0], -1, 3, 3)  # smplx is not that good an implementation, need to remove batch dimension
    _, A = batch_rigid_transform(Rs, tjoints, parents)  # here, parents should not have batch dimesion, constracdicting doc
    verts = forward_deform_lbs(verts, deform, weights, A, R, T)

    ret = dotdict()
    ret.verts = verts[0]
    ret.faces = tpose.faces
    export_dotdict(ret, args.output_file)


if __name__ == '__main__':
    main()
