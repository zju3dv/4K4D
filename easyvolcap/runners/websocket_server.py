"""
Create a websocket server during initialization and send the rendered images to the client
"""
from __future__ import annotations

import os
import glm
import torch
import asyncio
import threading
import websockets

import numpy as np
import torch.nn.functional as F
from typing import List, Union, Dict
from glm import vec3, vec4, mat3, mat4, mat4x3
from torchvision.io import decode_jpeg, encode_jpeg

from easyvolcap.engine import cfg  # need this for initialization?
from easyvolcap.engine import RUNNERS  # controls the optimization loop of a particular epoch
from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.viewer_utils import Camera
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.data_utils import add_iter, add_batch, to_cuda


@RUNNERS.register_module()
class WebSocketServer(VolumetricVideoViewer):
    # Viewer should be used in conjuction with another runner, which explicitly handles model loading
    def __init__(self,
                 # Runner related parameter & config
                 runner: VolumetricVideoRunner,  # already built outside of this init

                 # Socket related initialization
                 port: int = 1024,

                 # Camera related config
                 camera_cfg: dotdict = dotdict(type=Camera.__name__),
                 ):

        # Socket related initialization
        

        # Initialize server-side camera in case there's lag
        self.camera = Camera(**camera_cfg)

        # Runner initialization
        self.runner = runner
        self.runner.visualizer.store_alpha_channel = True  # disable alpha channel for rendering on viewer
        self.runner.visualizer.uncrop_output_images = False  # manual uncropping
        self.epoch = self.runner.load_network()  # load weights only (without optimizer states)
        self.dataset = self.runner.val_dataloader.dataset
        self.model = self.runner.model
        self.model.eval()

    def run(self):
        while True:
            image = self.render()  # H, W, 4, cuda gpu tensor

    def render(self):
        batch = self.camera.to_batch()
        batch = to_cuda(add_batch(batch))  # int -> tensor -> add batch -> cuda, smalle operations are much faster on cpu

        # Forward pass
        self.runner.maybe_jit_model(batch)
        with torch.inference_mode(self.runner.test_using_inference_mode), torch.no_grad(), torch.cuda.amp.autocast(enabled=self.runner.test_use_amp, cache_enabled=self.runner.test_amp_cached):
            output = self.model(batch)

        image = self.runner.visualizer.generate_type(output, batch, self.visualization_type)[0][0]  # RGBA (should we use alpha?)

        if self.exposure != 1.0 or self.offset != 0.0:
            image = torch.cat([(image[..., :3] * self.exposure + self.offset), image[..., -1:]], dim=-1)  # add manual correction
        image = (image.clip(0, 1) * 255).type(torch.uint8).flip(0)  # transform

        return image
