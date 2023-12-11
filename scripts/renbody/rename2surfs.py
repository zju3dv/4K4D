import os
from easyvolcap.utils.console_utils import *

skips = [
    'images',
    'images_calib',
    'maskes',
    'masks',
    'mask',
    'vhulls',
    'bgmtv2',
    'schp',
    'rvm',
    'bkgd',
    'mask_schp',
    'openpose',
    'heatmaps',
    'annots',
    'easymocap',
    'record',
    'result',
    'novel_view',
    'trained_model',
    'pafs',
    'output',
    'albedo',
    'shading',
    'smpl_params',
    'smpl_vertices',
    'params',
    'vertices',
    'dtu',
    'pointcloud_denoise',
]


def walk(d: str = 'data'):
    print(f'walking {os.path.abspath(d)}')
    cwd = os.getcwd()
    os.chdir(d)
    for i in os.listdir():
        if i == 'expanded_vhulls':
            os.system(f'mv expanded_vhulls surfs')
            break
        elif i in skips:
            continue
        elif os.path.isdir(i):
            walk(i)
    os.chdir(cwd)


walk()
