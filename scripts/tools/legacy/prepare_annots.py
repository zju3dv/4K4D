import sys
import json
import os
import glob
import numpy as np
import cv2
import argparse
from ruamel.yaml import YAML
yaml = YAML()


def get_cams():
    intri = cv2.FileStorage('intri.yml', cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage('extri.yml', cv2.FILE_STORAGE_READ)
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    for i in range(len(camera_names)):
        camera_name = camera_names[i]
        cams['K'].append(intri.getNode(f'K_{camera_name}').mat())
        cams['D'].append(
            intri.getNode(f'dist_{camera_name}').mat().T)
        cams['R'].append(extri.getNode(f'Rot_{camera_name}').mat())
        cams['T'].append(extri.getNode(f'T_{camera_name}').mat() * 1000)
    return cams


def get_img_paths():
    all_ims = []
    for i in range(len(camera_names)):
        camera_name = camera_names[i]
        cam_dir = f'{image_dir}/{camera_name}'
        ims = glob.glob(os.path.join(cam_dir, f'*{args.ext}'))
        ims = np.array(sorted(ims))
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


def get_kpts2d():
    def _get_kpts2d(paths):
        kpts2d_list = []
        for path in paths:
            with open(path, 'r') as f:
                d = json.load(f)
            kpts2d = np.array(d['people'][0]['pose_keypoints_2d']).reshape(
                -1, 3)
            kpts2d_list.append(kpts2d)
        kpts2d = np.array(kpts2d_list)
        return kpts2d

    all_kpts = []
    for i in range(len(camera_names)):
        camera_name = camera_names[i]
        cur_dump = f'keypoints2d/{camera_name}'
        paths = sorted(glob.glob(os.path.join(cur_dump, '*.json')))
        kpts2d = _get_kpts2d(paths[:1400])
        all_kpts.append(kpts2d)

    num_img = min([len(kpt) for kpt in all_kpts])
    all_kpts = [kpt[:num_img] for kpt in all_kpts]
    all_kpts = np.stack(all_kpts, axis=1)

    return all_kpts


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/subset_mi11')
parser.add_argument('--image_dir', type=str, default='images')
parser.add_argument('--humans', type=str, nargs='+', default=['talking_cont', 'talking_val', 'talking_step'])
parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
parser.add_argument('--use_existing_cam', action='store_true', default=False)
parser.add_argument('--use_existing_ims', action='store_true', default=False)
args = parser.parse_args()
camera_names = []
image_dir = args.image_dir
for human_ind in range(len(args.humans)):
    human = args.humans[human_ind]

    root = os.path.join(args.data_dir, human)
    old = os.getcwd()
    os.chdir(root)

    if args.use_existing_cam:
        existing = np.load('annots.npy', allow_pickle=True).item()
        camera_names = [
                f'Camera ({i+1})' for i in range(23)
            ]
        print(camera_names)
        cams = existing['cams']
    else:
        camera_names = yaml.load(open('extri.yml'))['names']
        print(camera_names)
        cams = get_cams()
    if args.use_existing_ims:
        img_paths = existing['ims']
    else:
        camera_names = sorted(os.listdir(args.image_dir))
        img_paths = get_img_paths()
    annot = {}
    annot['cams'] = cams

    ims = []
    for img_path in img_paths:
        data = {}
        data['ims'] = img_path.tolist()
        # data['kpts2d'] = kpt.tolist()  # TODO: inefficient but minimal code change
        ims.append(data)
    annot['ims'] = ims

    np.save('annots.npy', annot)
    # np.save('annots_python2.npy', annot, fix_imports=True)
    os.chdir(old)


"""
python tools/prepare_annots.py --data_dir data/my_zjumocap --humans my_313 my_315 my_377 my_386 my_387 my_390 my_392 my_393 my_394 
"""