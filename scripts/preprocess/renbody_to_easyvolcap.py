# This converts the RenBody dataset to easyvolcap format: storing images directly on disk as jpeg
# NOTE: This process could be extremely slow, as it needs to read the entire dataset into memory and process them accordingly
# NOTE: Saving raw images as jpeg is not recommended, as it will result in a loss of quality. However, this is the only way to maintain a reasonable level of storage cost

# Is it OK to directly read from these without preprocessing?
# It might be clumsy since we need to write a separate dataset module
# I'd rather perform some preprocessing and then save the data to disk as images

import os
import h5py
import torch
import inspect
import argparse
import numpy as np
from glob import glob
from os.path import join, dirname
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.viewer_utils import Camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera

# https://openxdlab.org.cn/details/RenBody

# We will
# 1. Find all cameras, sorted them as 12mp, 5mp, kinect
# 2. Read all images, mask & depth (if available) from the cameras, perform color correction on rgb images
#    1. Images will be saved on disk as jpegs (lossy, will use high quality options (100) but still lossy), will be color corrected before saving
#    2. Maskes will also be saved as 1-channel jpegs (0 or 255) (almost lossless if binary)
#    3. Depths will be saved raw as 16bit (uint16) pngs (lossless) (maybe use jp2 (lossy))
# 3. Convert camera formats to easyvolcap (H, W, K, R, T, D)

# * UPDATE: They stored jpeg bytes instead of raw pixels in the dataset
# We'd only need to store those on disk, but should we correct colors before storing?
# Maybe the camera format of easyvolcap requires a color correction matrix (like in D)

log(yellow('We\' omitting the Kinect camera for now, they look suspicious'))
camera_types = ['Camera_5mp', 'Camera_12mp']
images_dir = 'images'
masks_dir = 'masks'
# depths_dir = 'depths'


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renbody_root', type=str, default='/nas/datasets/human/RenBody/OpenXD-RenBody/partx')
    parser.add_argument('--easyvolcap_root', type=str, default='./data/renbody')
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--skip_camera', action='store_true', default=False)
    parser.add_argument('--skip_image', action='store_true', default=False)
    parser.add_argument('--skip_smplx', action='store_true', default=False)
    args = parser.parse_args()

    renbody_root = args.renbody_root
    easyvolcap_root = args.easyvolcap_root

    def process_scene(scene: str):
        scene_in_file = join(renbody_root, scene)
        scene_out_dir = join(easyvolcap_root, os.path.splitext(scene)[0])
        os.makedirs(scene_out_dir, exist_ok=True)

        dataset = h5py.File(scene_in_file)

        if not args.skip_camera: 
            cameras = dotdict()
            for camera_type in camera_types:
                for camera_id in dataset[camera_type]:  # renbody camera ids: 1, 2, 3, 4...
                    key = f'{int(camera_id):02d}'  # easyvolcap camera keys: 00, 01, 02...
                    cameras[key] = dotdict()  # might have more than 100 cameras?
                    cam = cameras[key]
                    cam.H = np.asarray(dataset[camera_type].attrs['resolution'][0])
                    cam.W = np.asarray(dataset[camera_type].attrs['resolution'][1])
                    cam.K = np.asarray(dataset['Calibration'][camera_type][camera_id]['K'])
                    cam.R = np.asarray(dataset['Calibration'][camera_type][camera_id]['R']).T  # 3, 3
                    cam.T = -cam.R @ np.asarray(dataset['Calibration'][camera_type][camera_id]['T'])  # 3, 3 @ 3
                    cam.D = np.asarray(dataset['Calibration'][camera_type][camera_id]['D'])
                    cam.ccm = np.asarray(dataset['Color_Calibration'][camera_id])

            write_camera(cameras, scene_out_dir)  # extri.yml and intri.yml
            log(yellow(f'Converted cameras saved to {blue(join(scene_out_dir, "{intri.yml,extri.yml}"))}'))

        def process_image(camera_type: str, camera_id: str, frame_id: str, img_dir: str, msk_dir: str):
            img_path = join(img_dir, f'{int(frame_id):06d}.jpg')  # renbody gives jpeg stream
            msk_path = join(msk_dir, f'{int(frame_id):06d}.jpg')  # renbody gives jpeg stream
            img = np.asarray(dataset[camera_type][camera_id]['color'][frame_id])  # This is heavy, read into memory
            msk = np.asarray(dataset[camera_type][camera_id]['mask'][frame_id])
            img.tofile(img_path)
            msk.tofile(msk_path)

        if not args.skip_image:
            params = dotdict()
            params.camera_type = []
            params.camera_id = []
            params.frame_id = []
            params.img_dir = []
            params.msk_dir = []
            for camera_type in camera_types:
                for camera_id in dataset[camera_type]:
                    img_dir = join(scene_out_dir, images_dir, f'{int(camera_id):02d}')
                    msk_dir = join(scene_out_dir, masks_dir, f'{int(camera_id):02d}')
                    os.makedirs(img_dir, exist_ok=True)
                    os.makedirs(msk_dir, exist_ok=True)
                    for frame_id in dataset[camera_type][camera_id]['color']:
                        params.camera_type.append(camera_type)
                        params.camera_id.append(camera_id)
                        params.frame_id.append(frame_id)
                        params.img_dir.append(img_dir)
                        params.msk_dir.append(msk_dir)

            parallel_execution(**params, action=process_image, sequential=False, print_progress=True)
            
        if not args.skip_smplx:
            params = dotdict()
            for key in dataset['SMPLx'].keys():
                params[key] = np.asarray(dataset['SMPLx'][key][()], dtype=np.float32)
            np.savez(join(scene_out_dir, 'motion.npz'), **params)
            # parallel_execution(**params, action=process_smplx, sequential=False, print_progress=True)

    scenes = os.listdir(renbody_root)
    scenes = sorted(scenes)
    if args.scene is not None:
        log(yellow(f'Only process scene: {args.scene}'))
        scenes = [x for x in scenes if args.scene in x]
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)  # literally a for-loop


if __name__ == '__main__':
    main()

"""
{
    "atrrs"                                 : 动作序列属性,
    "Images"                                : 多视角图片,
    "Calibration"                           : 相机标定参数,
    "Annotation"                            : 标注参数,
}

{
    ".attrs"{
        "age"                               : int,
        "gender"                            : str ["male", "female"],
        "height"                            : int (厘米),
        "weight"                            : int (公斤),
    }
    "Camera_12mp.attrs"{
        "num_device"                        : int,
        "num_frame"                         : int,
        "resolution"                        : int (高, 宽),
    }
    "Camera_5mp.attrs"{
        ...                                 : 与"Camera_12mp"同结构
    }
    "Kinect.attrs"{
        ...                                 : 与"Camera_12mp"同结构
    }
}

{
    "Camera_12mp"{
        "cameraID"{
            "color"{
                "frameID"                   : uint8 (4096,3000,3),
                ...
            },
            "mask"{
                "frameID"                   : uint8 (4096,3000),
            }
        }
    }
    "Camera_5 mp"{
        ...                                 : 与"Camera_12mp"同结构
    }
    "Kinect"{                               : Kinect组,
        "cameraID"{
            "depth"{
                "frameID"                   : uint16 (576,640),
                ...
            },
            "mask"{
                "frameID"                   : uint8 (576,640),
            }
        }
    }
}

{
    "Camera_12mp"{
        "cameraID"{                         : str,
            "K"                             : double (3,3) 内参矩阵,
            "D"                             : double (5,) 畸变系数,
            "RT"                            : double (4,4) 相机到世界坐标系变换
        }
    }
    "Camera_5mp"{
        ...                                 : 与"Camera_12mp"同结构
    }
    "Kinect"{
        ...                                 : 与"Camera_12mp"同结构
    }
}

{
    "Color_Calibration"{
        "cameraID"                          : double (3,3) "cameraID"的BGR颜色的矫正系数,
        ...
    }
    "Keypoint2d"{
        "frameID"                           : double (frame_num,joint_num,3)
        ...
    }
    "Keypoints3d"{
        "frameID"                           : double (frame_num,joint_num,3)
        ...
    }
    "SMPLx"{
        "frameID"{
            "beta"                          : double (1,10) 演员体型参数
            "body_pose"                     : double (frame_num,66)逐帧动作参数
            ...                             : 其他smplx的参数
        }
        ...
    }
}
"""
