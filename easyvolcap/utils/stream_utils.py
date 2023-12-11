import cv2
import time
import torch
from os.path import join
import torch.nn.functional as F
from multiprocessing import Process, Queue
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *


class MultiWebcamUSB:
    # Helper class to manage camera parameters
    def __init__(self,
                 cam_cfgs=dotdict(),
                 save_dir=f'data/webcam/simple/actor1',
                 save_tag=f'images'):
        # Camera parameters and save directory
        self.cam_cfgs = cam_cfgs
        self.save_dir = save_dir
        self.save_tag = save_tag
        self.save_pth = join(self.save_dir, self.save_tag)
        self.num_cams = len(self.cam_cfgs)
        self.count = 0

        # Shared queue to store images
        self.max_size = self.num_cams * 3
        self.queue_imgs = Queue(maxsize=self.max_size * 2)  # keep 0.1s
        self.queue_flag = Queue(maxsize=3)

        # Camera threads
        self.camera_threads = []
        for i, cam_cfg in enumerate(self.cam_cfgs.values()):
            thread = Process(target=self.start, args=(i, cam_cfg, self.queue_imgs, self.queue_flag))
            thread.start()
            self.camera_threads.append(thread)

    def start(self, idx: int, cam_cfg: dotdict,
              queue_imgs: Queue, queue_flag: Queue):
        # Open camera and set parameters
        cap = cv2.VideoCapture(cam_cfg.index)
        assert cap.isOpened(), log(red(f"Cannot open camera {cam_cfg.index}."))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Set resolution and fps if specified
        if 'shape' in cam_cfg:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.shape[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.shape[1])
        if 'fps' in cam_cfg:
            cap.set(cv2.CAP_PROP_FPS, cam_cfg.fps)

        # Read like say, 25 frames to warm up the camera
        for _ in range(25): cap.read()

        # Start capturing images
        while queue_flag.qsize() == 0:
            # Capture image from the camera
            while queue_imgs.qsize() > self.max_size: queue_imgs.get()
            ret, img = cap.read()
            if not ret:
                for _ in range(10):
                    time.sleep(0.01)
                    ret, img = cap.read()
                    if ret: break
                else: break
            tsp = int(time.time() * 1000)
            queue_imgs.put(dotdict({'idx': idx, 'tsp': tsp, 'img': img}))

        # Release the camera
        cap.release()

    def capture(self, save=False):
        rets, flag = dotdict(), False
        while not flag:
            data = self.queue_imgs.get()
            rets[data.idx] = data
            tsps = [data.tsp for data in rets.values()]
            flag = len(tsps) == self.num_cams
            flag = flag and (max(tsps) - min(tsps) < 40)

        # Return images
        imgs = [np.asarray(cv2.cvtColor(rets[i].img, cv2.COLOR_BGR2RGB), dtype=np.float32) for i in range(self.num_cams)]

        # Save images
        # TODO: whether it is a good idea to save images here
        if save:
            for i, img in enumerate(imgs):
                save_pth = f'{self.save_pth}/{i}'
                if not os.path.exists(save_pth): os.makedirs(save_pth, exist_ok=True)
                cv2.imwrite(f'{save_pth}/{self.count:06d}.jpg', img[..., [2, 1, 0]])
            self.count += 1

        return imgs

    def __del__(self):
        self.queue_flag.put(True)
        for thread in self.camera_threads:
            thread.join()
