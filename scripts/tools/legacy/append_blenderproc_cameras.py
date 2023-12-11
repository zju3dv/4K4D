import os
import argparse
import numpy as np

import json
from os.path import join

# fmt: off
import sys
sys.path.append('.')

from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.console_utils import log
# fmt: on

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', default='data/synthetic_human')
    parser.add_argument('--append_root', default='/nas/home/gengchen/develop/blenderproc_render/output_old')
    args = parser.parse_args()
    
    for human in os.listdir(args.input_root):
        human_root = join(args.input_root, human)
        proc_root = join(args.append_root, f'{human}_', 'test/config')
        if not os.path.exists(proc_root):
            log(f'{proc_root} does not exist, skipping', 'yellow')
            continue
        
        cameras = read_camera(join(human_root, 'intri.yml'), join(human_root, 'extri.yml'))
        if 'basenames' in cameras:
            del cameras['basenames']
        n_previous = len(cameras)
        to_append = sorted(os.listdir(proc_root))
        
        for frame_cam in to_append:
            data = json.load(open(join(proc_root, frame_cam)))
            K = np.array(data['K'])
            RT = np.array(data['RT'])
            R = RT[:3, :3]
            T = RT[:3, 3:]
            
            name = f'{n_previous + int(os.path.splitext(frame_cam)[0].split("_")[-1]):02d}'
            cameras[name] = {
                'K': K,
                'R': R,
                'T': T,
            }
        log(f'{len(cameras)} saved to {human_root}')
        # write_camera(cameras, join(human_root, 'all_cameras'))
        write_camera(cameras, human_root)
        
        
if __name__ == '__main__':
    main()