import torch
import argparse
import numpy as np

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.data_utils import export_mesh, load_mesh
from easyvolcap.utils.mesh_utils import laplacian_smoothing, hierarchical_winding_distance_remesh, get_edges, adjacency, winding_number_nooom, segment_mesh, bidirectional_icp_fitting, loop_subdivision
from easyvolcap.utils.sem_utils import semantic_dim, semantic_list
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clothed_input', default='data/xuzhen36/talk/registration/deformation/semantic_mesh.npz')
    parser.add_argument('--body_input', default='data/xuzhen36/talk/registration/deformation/semantic_smplh.npz')
    parser.add_argument('--body_output', default='data/xuzhen36/talk/registration/deformation/body_mesh.ply')
    parser.add_argument('--cloth_output', default='data/xuzhen36/talk/registration/deformation/cloth_mesh.ply')
    parser.add_argument('--cloth_list', nargs='+', default=['upper_cloth'])
    args = parser.parse_args()

    # global arguments
    device = 'cuda'

    # maybe perform subdivision before hand? and use catmull clark instead of simple subdivision provided by trimesh
    # https://onrendering.com/data/papers/catmark/HalfedgeCatmullClark.pdf
    # https://github.com/jdupuy/HalfedgeCatmullClark
    v0, f0 = load_mesh(args.clothed_input, device)
    vs0 = torch.tensor(np.load(args.clothed_input)['verts_semantics'], device=v0.device)
    i1 = list(map(lambda x: semantic_list.index(x), args.cloth_list))
    i1 = torch.tensor(i1, device=v0.device, dtype=torch.long)

    # segment based on vertex semantices
    v, f = segment_mesh(v0, f0, vs0, i1, smoothing='edge')
    v, f = loop_subdivision(v, f, 1)

    # save the results
    export_mesh(v, f, filename=args.cloth_output)

    # extract body mesh
    i0 = list(map(lambda x: semantic_list.index(x), [s for s in semantic_list if s not in args.cloth_list]))
    i0 = torch.tensor(i0, device=v.device, dtype=torch.long)
    v0, f0 = segment_mesh(v0, f0, vs0, i0, smoothing='edge', dilate=-1)
    v0, f0 = loop_subdivision(v0, f0, 1)

    v1, f1 = load_mesh(args.body_input, device)
    vs1 = torch.tensor(np.load(args.body_input)['verts_semantics'], device=v.device)
    v1, f1 = segment_mesh(v1, f1, vs1, i1, smoothing='edge', dilate=3)
    v1, f1 = loop_subdivision(v1, f1, 2)

    v0, v1 = bidirectional_icp_fitting(v0, f0, v1, f1)

    level_set = 0.334

    v2, f2 = torch.cat([v0, v1]), torch.cat([f0, f1+len(v0)])
    # v, f = v2, f2
    v, f = hierarchical_winding_distance_remesh(v2, f2, level_set=level_set)
    # 0.334 will produce ripple effects on perfectly normal mesh if thresh of winding number is too low
    # (1 - th) / 2 + (level_set - 0.5).abs() > 0.5 - maximum_error
    # 0.5 - th / 2 + 0.5 - level_set - 0.5 > - maximum_error
    # th / 2 + level_set - 0.5 > maximum_error
    # 0.225 + 0.334 - 0.5 = 0.059 > maximum_error

    # conditional laplacian smoothing
    th = 0.45  # 1-max_error
    wn = winding_number_nooom(v, v2, f2)  # TODO: use grid sample from previous computation to make this faster
    vm = (wn - level_set).abs() < (th / 2)

    # compute edges
    e, i, c = get_edges(f)

    # dialte the winding number thresh selection
    dilate = 15  # much thicker mesh
    A = adjacency(v, e)
    vm = vm.float()
    for i in range(abs(dilate)):
        vm = A @ vm
    vm = vm.bool()
    svi = vm.nonzero(as_tuple=True)[0]

    v = laplacian_smoothing(v, e, svi, iter=20)

    # save the results
    export_mesh(v, f, filename=args.body_output)
    
    
    # TODO:
    # 1. extract blend weights from closest SMPL points? or just from the original mesh? (whose blend weights are extracted using blender)
    # 2. extract a good enough UV parameterization (will need to first compute a good enough segmentation): NOTE: would require a rest pose (and t-pose is not a rest pose... for cloth)


if __name__ == '__main__':
    main()
