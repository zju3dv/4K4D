import zlib
import torch
import asyncio
import websockets
from torchvision.io import encode_jpeg, decode_jpeg

import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.viewer_utils import Camera

camera = Camera(H=1080, W=1920)
uri = "ws://10.76.5.252:1024"


async def websocket_client():
    async with websockets.connect(uri) as websocket:
        frame_cnt = 0
        prev_time = time.perf_counter()

        while True:
            buffer = await websocket.recv()
            # img = jpeg.decode(response)  # H, W, 4
            img = decode_jpeg(torch.from_numpy(np.frombuffer(buffer, np.uint8)), device='cuda')

            camera_data = zlib.compress(camera.to_string().encode('ascii'))
            await websocket.send(camera_data)

            curr_time = time.perf_counter()
            pass_time = curr_time - prev_time
            frame_cnt += 1
            if pass_time > 2.0:
                fps = frame_cnt / pass_time
                frame_cnt = 0
                prev_time = curr_time
                log(f'Receive FPS: {fps}')

asyncio.get_event_loop().run_until_complete(websocket_client())
asyncio.get_event_loop().run_forever()
