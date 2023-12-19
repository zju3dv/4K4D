# Converts raw genebody dataset format to easyvolcap
# Links images and maybe convert the whole folder

import os
import cv2
import argparse
import numpy as np
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genebody_root', type=str, default='~/datasets/genebody_test10')
    parser.add_argument('--easyvolcap_root', type=str, default='./data/genebody_test10')
    args = parser.parse_args()

    genebody_root = args.genebody_root
    easyvolcap_root = args.easyvolcap_root

    # clear, NOTE: careful
    # os.system(f'rm -rf {easyvolcap_root}')

    def process_scene(scene):
        # A standard mulit-view scene
        cam_out_dir = join(easyvolcap_root, scene)
        os.makedirs(cam_out_dir, exist_ok=True)

        # Load and process camera parameters
        annots = np.load(join(genebody_root, f'{scene}/annots.npy'), allow_pickle=True).item()['cams']
        cams = dotdict()
        cams_name = sorted(os.listdir(join(genebody_root, f'{scene}/image')))
        for i, cam_name in enumerate(cams_name):
            cam = dotdict()
            cam.K = annots[cam_name]['K']
            w2c = np.linalg.inv(annots[cam_name]['c2w'])
            cam.R = w2c[:3, :3]
            cam.T = w2c[:3, 3:]
            cam.D = annots[cam_name]['D'].reshape(-1, 1)
            cams[f'{i:02d}'] = cam
        # Store camera parameters
        write_camera(cams, cam_out_dir)
        log(yellow(f'Converted cameras saved to {blue(join(cam_out_dir, "{intri.yml,extri.yml}"))}'))

        def process_image(rmap_cam_idx, orig_cam_idx):
            # Define input directories
            img_in_dir = join(genebody_root, f'{scene}/image/{orig_cam_idx}')
            msk_in_dir = join(genebody_root, f'{scene}/mask/{orig_cam_idx}')

            # Define and create output directories
            img_out_dir = join(easyvolcap_root, f'{scene}/images/{rmap_cam_idx}')
            msk_out_dir = join(easyvolcap_root, f'{scene}/masks/{rmap_cam_idx}')
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(msk_out_dir, exist_ok=True)
            
            # Link images
            for j, img_name in enumerate(sorted(os.listdir(img_in_dir))):
                img_in_path = join(img_in_dir, img_name)
                # Writing and linking
                img_out_path = join(img_out_dir, f'{j:06d}.jpg')
                os.system(f'ln -s {img_in_path} {img_out_path}')

            # Link masks
            for j, msk_name in enumerate(sorted(os.listdir(msk_in_dir))):
                msk_in_path = join(msk_in_dir, msk_name)
                # Writing and linking
                msk_out_path = join(msk_out_dir, f'{j:06d}.png')
                os.system(f'ln -s {msk_in_path} {msk_out_path}')

        # Process smpl/smplx parameters and vertices
        smpl_params_in_dir = join(genebody_root, f'{scene}/param')
        smpl_vertex_in_dir = join(genebody_root, f'{scene}/smpl')

        # NOTE: some scene does not have smplx vertices, eg. `joseph_matanda`
        if os.path.exists(smpl_params_in_dir) and os.path.exists(smpl_vertex_in_dir):
            # Define and create smpl related output directories
            smpl_params_out_dir = join(easyvolcap_root, f'{scene}/smpl_params')
            smpl_vertex_out_dir = join(easyvolcap_root, f'{scene}/smpl_vertices')
            os.makedirs(smpl_params_out_dir, exist_ok=True)
            os.makedirs(smpl_vertex_out_dir, exist_ok=True)

            for i, param_name in enumerate(sorted(os.listdir(smpl_params_in_dir))):
                params_in_path = join(smpl_params_in_dir, param_name)
                vertex_in_path = join(smpl_vertex_in_dir, param_name.replace('.npy', '.obj'))
                
                # FIXME: I'm not sure whether `params['smplx_scale]` has any effect on the final result
                gebody_params = dotdict(np.load(params_in_path, allow_pickle=True).item()['smplx'])
                vertices, faces = load_mesh(vertex_in_path, device='cpu')
                vertices, faces = vertices.numpy().astype(np.float32), faces.numpy().astype(np.float32)
                
                # Writing the smpl vertices
                vertex_out_path = join(smpl_vertex_out_dir, f'{i}.npy')
                np.save(vertex_out_path, vertices)
                
                # Processing the smpl parameters
                easyvv_params = dotdict()
                easyvv_params.pose = gebody_params.body_pose.reshape(1, -1).astype(np.float32)
                easyvv_params.Rh = gebody_params.global_orient.astype(np.float32)
                easyvv_params.Th = gebody_params.transl.astype(np.float32)
                easyvv_params.shapes = gebody_params.betas.astype(np.float32)
                params_out_path = join(smpl_params_out_dir, f'{i}.npy')
                np.save(params_out_path, easyvv_params)

        # TODO: add smpl_lbs info processing if needed in the future
        # see `data/my_zjumocap/my_131/smpl_lbs` for example

        parallel_execution([f'{i:02d}' for i in range(len(cams_name))], cams_name, action=process_image)

    scenes = os.listdir(genebody_root)
    scenes = sorted(scenes)
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)


if __name__ == '__main__':
    main()
