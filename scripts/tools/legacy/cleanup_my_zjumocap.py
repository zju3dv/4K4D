# This script removes the vertices folder
# and loads camera parameters from annots.npy
# for the copied my_zjumocap dataset used in the original enerf and stableenerf implementation

import os
import numpy as np
import subprocess
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera

data_root = 'data/my_zjumocap'
for human in os.listdir(data_root):
    human_root = join(data_root, human)
    a = np.load(f'{human_root}/annots.npy', allow_pickle=True).item()
    cams = a['cams']
    cams_dict = {}
    for i in range(len(cams['K'])):
        key = f'{i:02d}'
        cams_dict[key] = {}
        cams_dict[key]['K'] = np.asarray(cams['K'][i])
        cams_dict[key]['R'] = np.asarray(cams['R'][i])
        cams_dict[key]['T'] = np.asarray(cams['T'][i]) / 1000
        cams_dict[key]['D'] = np.asarray(cams['D'][i])

    write_camera(cams_dict, human_root)