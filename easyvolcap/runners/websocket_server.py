"""
Create a websocket server during initialization and send the rendered images to the client
"""
from __future__ import annotations

import os
import glm
import time
import zlib
import torch
import asyncio
import threading
import websockets
import numpy as np
import torch.nn.functional as F

from copy import deepcopy
from typing import List, Union, Dict
from glm import vec3, vec4, mat3, mat4, mat4x3
from torchvision.io import encode_jpeg, decode_jpeg

from easyvolcap.engine import cfg  # need this for initialization?
from easyvolcap.engine import RUNNERS  # controls the optimization loop of a particular epoch
from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.viewer_utils import Camera
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.data_utils import add_iter, add_batch, to_cuda, Visualization


@RUNNERS.register_module()
class WebSocketServer:
    # Viewer should be used in conjuction with another runner, which explicitly handles model loading
    def __init__(self,
                 # Runner related parameter & config
                 runner: VolumetricVideoRunner,  # already built outside of this init

                 # Socket related initialization
                 host: str = '0.0.0.0',
                 port: int = 1024,

                 # Camera related config
                 camera_cfg: dotdict = dotdict(),
                 jpeg_quality: int = 75,
                 window_size: List[int] = [768, 1366],

                 **kwargs,
                 ):

        # Socket related initialization
        self.host = host
        self.port = port

        # Initialize server-side camera in case there's lag
        self.camera_cfg = camera_cfg
        self.camera = Camera(**camera_cfg)
        self.H, self.W = window_size
        self.image = torch.randint(0, 255, (self.H, self.W, 4), dtype=torch.uint8)
        self.lock = threading.Lock()
        self.stream = torch.cuda.Stream()
        self.jpeg_quality = jpeg_quality

        # Runner initialization
        self.runner = runner
        self.runner.visualizer.store_alpha_channel = True  # disable alpha channel for rendering on viewer
        self.runner.visualizer.uncrop_output_images = True  # pass the whole image to viewer, TODO: make this more robust, let server pass info to viewer
        self.visualization_type = Visualization.RENDER
        self.epoch = self.runner.load_network()  # load weights only (without optimizer states)
        self.iter = self.epoch * self.runner.ep_iter  # loaded iter
        self.dataset = self.runner.val_dataloader.dataset
        self.model = self.runner.model
        self.model.eval()

    def run(self):
        def start_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            log('Preparing websocket server for sending images & receiving cameras')
            server = websockets.serve(self.server_loop, self.host, self.port)

            loop.run_until_complete(server)
            loop.run_forever()

        self.server_thread = threading.Thread(target=start_server, daemon=True)
        self.server_thread.start()
        self.render_loop()  # the rendering runs on the main thread

    @property
    def H(self): return self.camera.H

    @property
    def W(self): return self.camera.W

    @H.setter
    def H(self, value): self.camera.H = value

    @W.setter
    def W(self, value): self.camera.W = value

    def render_loop(self):  # this is the main thread
        frame_cnt = 0
        prev_time = time.perf_counter()

        while True:
            batch = self.camera.to_batch()  # fast copy of camera parameter
            image = self.render(batch)  # H, W, 4, cuda gpu tensor
            self.stream.wait_stream(torch.cuda.current_stream())  # initiate copy after main stream has finished
            with torch.cuda.stream(self.stream):
                with self.lock:
                    self.image = image.to('cpu', non_blocking=True)  # initiate async copy

            curr_time = time.perf_counter()
            pass_time = curr_time - prev_time
            frame_cnt += 1
            if pass_time > 2.0:
                fps = frame_cnt / pass_time
                frame_cnt = 0
                prev_time = curr_time
                log('Renderer FPS:', fps)
                log('Renderer camera shape:', self.camera.H, self.camera.W)
                log('Renderer image sum:', self.image.sum())

    async def server_loop(self, websocket: websockets.WebSocket, path: str):
        frame_cnt = 0
        prev_time = time.perf_counter()

        while True:
            self.stream.synchronize()  # waiting for the copy event to complete
            with self.lock:
                image = self.image.numpy()  # copy to new memory space
            image = encode_jpeg(torch.from_numpy(image).permute(2, 0, 1), quality=self.jpeg_quality).numpy().tobytes()
            await websocket.send(image)

            response = await websocket.recv()
            if len(response):
                camera = Camera()
                camera.from_string(zlib.decompress(response).decode('ascii'))
                self.camera = camera

            curr_time = time.perf_counter()
            pass_time = curr_time - prev_time
            frame_cnt += 1
            if pass_time > 2.0:
                fps = frame_cnt / pass_time
                frame_cnt = 0
                prev_time = curr_time
                log('Server FPS:', fps)
                log('Server camera shape:', self.camera.H, self.camera.W)
                log('Server image sum:', self.image.sum())

    def render(self, batch: dotdict):
        batch = self.dataset.get_viewer_batch(batch)
        batch = to_cuda(add_batch(add_iter(batch, self.iter, self.runner.total_iter)))

        # Forward pass
        self.runner.maybe_jit_model(batch)
        with torch.inference_mode(self.runner.test_using_inference_mode), torch.no_grad(), torch.cuda.amp.autocast(enabled=self.runner.test_use_amp, cache_enabled=self.runner.test_amp_cached):
            output = self.model(batch)

        image = self.runner.visualizer.generate_type(output, batch, self.visualization_type)[0][0]  # RGBA (should we use alpha?)
        image = image[..., :3]
        image = (image.clip(0, 1) * 255).type(torch.uint8).flip(0)  # transform
        return image  # H, W, 3
