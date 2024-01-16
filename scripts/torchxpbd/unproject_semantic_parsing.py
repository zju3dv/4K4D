import os
import cv2
import torch
import kornia
import argparse
import numpy as np
import nvdiffrast.torch as dr

from termcolor import cprint
from torch_scatter import scatter

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.chunk_utils import linear_gather
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.raster_utils import get_ndc_perspective_matrix
from easyvolcap.utils.sem_utils import color_to_semantic, semantics_to_color, semantic_list, semantic_dim, get_schp_palette
from easyvolcap.utils.data_utils import load_image, list_to_tensor, export_mesh, to_cuda, export_dotdict

"""
Task: unproject the values of the schp semantic parsing onto the mesh vertices to get a general result
Requirements:
1. use integer values for a clear segmentation
2. take care of mesh edges: i.e. build a precedence over background
3. store a semantic mask as vertex property instead of a semantic texture for uv (might be problematic?)

Procedure:
1. read the mesh file (along with blend weights, vertex colors)
2. read the schp semantic parsing and corresponding camera parameters (easymocap format, generated from blender)
3. perform forward rasterization to get pixel and face correspondence
4. assign semantic value to every vertices seen by faces, use median to determine the final output (or just max for simplicity?)
5. output a viewable ply mesh, whose vertex color are assigned the corresponding palette value

NOTE: this file heavily relies on the output of my idr-pipeline output, which can get a little bit messy since it hasn't been made into a well-documented api
NOTE: moreover, there exits dependency on the actual utility function that I wrote during the years
"""


def rasterize(verts: torch.Tensor,  # B, V, 3
              faces: torch.Tensor,  # F, 3: no batch
              R: torch.Tensor,  # B, 3, 3: w2c
              T: torch.Tensor,  # B, 3, 1: w2c
              K: torch.Tensor,  # B, 4, 4: perspective (opencv)
              H: int,  # no batch
              W: int,  # no batch
              ):

    vverts = verts @ R.mT + T.mT
    padding = vverts.new_ones((*vverts.shape[:-1], 1))  # w -> 1
    homovverts = torch.cat([vverts, padding], dim=-1)
    K = get_ndc_perspective_matrix(K, H, W)  # B, 4, 4 -> B, 4, 4 or B, 3, 3 -> B, 4, 4
    ndcverts = homovverts @ K.mT
    drfaces = faces.int()
    rast, _ = dr.rasterize(glctx, ndcverts, drfaces, resolution=[H, W])

    # The first output tensor has shape [minibatch_size, height, width, 4] and contains the main rasterizer output in order (u, v, z/w, triangle_id).
    return rast


def assign_to_vertices(face_ids: torch.Tensor,  # B, H, W
                       sem_msk: torch.Tensor,  # B, H, W
                       faces: torch.Tensor,  # F, 3
                       ):
    # Core of the whole algorithm
    # Assign appropriate semantic class to every vertices
    # Currently we rely on the max reduction of torch_scatter to make this seem like an easy task
    # But I do think this needs a little bit more consideration, maybe checkout how boyi handled this?
    # Did he just performed optimization?

    # We used an O(n_class) algorithm to determine the max appearance of a particular class
    # TODO: definitely not possible to do this for different pixel colors

    F = faces.shape[0]
    V = faces.max().item() + 1  # number of vertices # MARK: sync
    face_ids = face_ids.view(-1)
    sem_msk = sem_msk.view(-1)
    valid_inds = (face_ids != -1).nonzero(as_tuple=True)[0]
    face_ids = linear_gather(face_ids, valid_inds)  # VALID
    sem_msk = linear_gather(sem_msk, valid_inds)  # VALID

    # We want a 20, V tensor to store all occurrences of the semantic classes
    faces_semantics = []
    verts_semantics = []
    for i in range(semantic_dim):
        faces_semantics_curr = scatter((sem_msk == i).int(), face_ids, dim_size=F, reduce='sum')  # F,
        verts_semantics_curr = scatter(faces_semantics_curr[..., None].expand(-1, 3).reshape(-1), faces.view(-1), dim_size=V, reduce='sum')  # V,
        faces_semantics.append(faces_semantics_curr)
        verts_semantics.append(verts_semantics_curr)
    verts_semantics = torch.stack(verts_semantics, dim=0)  # 20, V
    verts_semantics = torch.argmax(verts_semantics, dim=0)  # V,

    faces_semantics = torch.stack(faces_semantics, dim=0)  # 20, V
    faces_semantics = torch.argmax(faces_semantics, dim=0)  # V,

    return faces_semantics, verts_semantics


def get_cameras(intri_path, extri_path, camera_names):
    # Read camera parameters as float32 tensor array
    intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    cams = dotdict(cams)
    for i in range(len(camera_names)):
        camera_name = camera_names[i]
        cams['K'].append(intri.getNode(f'K_{camera_name}').mat())
        cams['D'].append(intri.getNode(f'dist_{camera_name}').mat())
        cams['R'].append(extri.getNode(f'Rot_{camera_name}').mat())
        cams['T'].append(extri.getNode(f'T_{camera_name}').mat())
    for k in cams:
        cams[k] = torch.from_numpy(np.stack(cams[k]).astype(np.float32)).to('cuda', non_blocking=True)
    return cams


def main():
    # Commandline argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default='data/xuzhen36/talk/registration/deformation/optim_mesh.npz')  # a full mesh with verts and faces, blend weights and stuff
    parser.add_argument('-o', '--output_file', default='data/xuzhen36/talk/registration/deformation/semantic_mesh.npz')
    parser.add_argument('--data_dir', default='data/xuzhen36/talk/circular_ply/xuzhen_near2_01')  # FIXME: xy dependency for circular rendering (actual problem here cuz these code aren't mine)
    parser.add_argument('--schp_dir', default='schp')
    parser.add_argument('--image_dir', default='images')
    parser.add_argument('--image', default='000000.png')
    args = parser.parse_args()

    # Read cameras from the data_dir: intri.yml and extri.yml
    camera_names = sorted(os.listdir(f'{args.data_dir}/{args.schp_dir}'))
    cameras = get_cameras(f'{args.data_dir}/intri.yml', f'{args.data_dir}/extri.yml', camera_names)

    # Load the SCHP color images, apply undistortion (on GPU) and convert colored image to semantic class id
    image_list = [f'{args.data_dir}/{args.schp_dir}/{camera_name}/{args.image}' for camera_name in camera_names]
    schp = list_to_tensor(parallel_execution(image_list, action=load_image))  # B, C, H, W
    schp = kornia.geometry.calibration.undistort_image(schp, cameras.K, cameras.D[..., 0])  # remove last dim on D
    schp = (schp.permute(0, 2, 3, 1) * 255 + 0.5).to(torch.uint8)

    palette = get_schp_palette(semantic_dim)
    max_len = max(list(map(len, semantic_list)))
    for i, sem in enumerate(semantic_list):
        bg_color = palette[i]
        if bg_color.sum() > 3 * 128:
            fg_color = [0, 0, 0]
        else:
            fg_color = [255, 255, 255]
        content = f'{sem}: {i}'
        # Print the actual colored used in the SCHP segmentation illustration for better understanding
        # print(colored_rgb(fg_color, bg_color, f'{content: <{max_len+5}}'))

    palette = torch.from_numpy(palette).to('cuda', non_blocking=True)
    sem_msk = color_to_semantic(schp, palette)
    B, H, W = sem_msk.shape

    # Load mesh data: vertices related and some other LBS related parameters
    tpose = np.load(args.input_file)
    tpose = dotdict(**tpose)
    tpose = to_cuda(tpose)

    faces = tpose.faces
    verts = tpose.verts[None].expand(B, *tpose.verts.shape)

    # Perform rasterization to get face ids for every corresponding pixels of all cameras
    rast: torch.Tensor = rasterize(verts, faces, cameras.R, cameras.T, cameras.K, H, W)

    # Voting for vertices' attributes (semantics in this case)
    faces_semantics, verts_semantics = assign_to_vertices(rast[..., 3].long() - 1, sem_msk, faces)
    colors = semantics_to_color(verts_semantics.long(), palette)

    mesh_output_file = args.output_file.replace('.npz', '.ply')
    cprint(f'saving mesh visualization to: {mesh_output_file}', color='blue')
    export_mesh(verts[0], faces, colors=colors, filename=mesh_output_file)

    tpose.faces_semantics = faces_semantics
    tpose.verts_semantics = verts_semantics
    tpose.posed_verts = verts[0]
    cprint(f'saving mesh package to: {args.output_file}', color='blue')
    export_dotdict(tpose, args.output_file)


if __name__ == '__main__':
    glctx = dr.RasterizeGLContext(output_db=False)
    main()
