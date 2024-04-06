import cv2
import zlib
import torch
import asyncio
import threading
import websockets
import numpy as np
from copy import deepcopy
from torchvision.io import encode_jpeg, decode_jpeg

from glm import vec3, vec4, mat3, mat4, mat4x3
from imgui_bundle import imgui_color_text_edit as ed
from imgui_bundle import portable_file_dialogs as pfd
from imgui_bundle import imgui, imguizmo, imgui_toggle, immvision, implot, ImVec2, ImVec4, imgui_md, immapp, hello_imgui

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.viewer_utils import Camera, CameraPath
from easyvolcap.utils.data_utils import add_iter, add_batch, to_cuda, Visualization

# fmt: off
from easyvolcap.engine import cfg, args
from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer
import glfw
# fmt: on


class Viewer(VolumetricVideoViewer):
    def __init__(self,
                 window_size: List[int] = [768, 1366],  # height, width
                 window_title: str = f'EasyVolcap WebSocket Client',  # MARK: global config

                 font_size: int = 18,
                 font_bold: str = 'assets/fonts/CascadiaCodePL-Bold.otf',
                 font_italic: str = 'assets/fonts/CascadiaCodePL-Italic.otf',
                 font_default: str = 'assets/fonts/CascadiaCodePL-Regular.otf',
                 icon_file: str = 'assets/imgs/easyvolcap.png',

                 use_quad_cuda: bool = True,
                 use_quad_draw: bool = False,
                 fullscreen: bool = False,
                 compose: bool = False,
                 compose_power: float = 1.0,

                 update_fps_time: float = 0.5,  # be less stressful
                 update_mem_time: float = 0.5,  # be less stressful

                 render_meshes: bool = True,
                 render_network: bool = True,

                 camera_cfg: dotdict = dotdict(type=Camera.__name__, string='{"H":768,"W":1366,"K":[[736.5288696289062,0.0,682.7473754882812],[0.0,736.4380493164062,511.99737548828125],[0.0,0.0,1.0]],"R":[[0.9938720464706421,0.0,-0.11053764075040817],[-0.0008741595083847642,0.9999688267707825,-0.007859790697693825],[0.1105341762304306,0.007908252067863941,0.9938408732414246]],"T":[[-0.2975313067436218],[-1.2581647634506226],[0.2818146347999573]],"n":4.0,"f":2000.0,"t":0.0,"v":0.0,"bounds":[[-20.0,-15.0,4.0],[20.0,15.0,25.0]],"mass":0.10000000149011612,"moment_of_inertia":0.10000000149011612,"movement_force":10.0,"movement_torque":1.0,"movement_speed":10.0,"origin":[0.0,0.0,0.0],"world_up":[0.0,-1.0,0.0]}'),

                 ):
        # Camera related configurations
        self.camera_cfg = camera_cfg
        self.fullscreen = fullscreen
        self.window_size = window_size
        self.window_title = window_title

        # Quad related configurations
        self.use_quad_draw = use_quad_draw
        self.use_quad_cuda = use_quad_cuda
        self.compose = compose
        self.compose_power = compose_power

        # Font related config
        self.font_default = font_default
        self.font_italic = font_italic
        self.font_bold = font_bold
        self.font_size = font_size
        self.icon_file = icon_file

        self.render_meshes = render_meshes
        self.render_network = render_network

        self.update_fps_time = update_fps_time
        self.update_mem_time = update_mem_time

        # Timinigs
        self.acc_time = 0
        self.prev_time = 0
        self.playing = False
        self.playing_fps = 30
        self.playing_speed = 0.005
        self.discrete_t = False
        self.use_vsync = False

        self.init_camera(camera_cfg)  # prepare for the actual rendering now, needs dataset -> needs runner
        self.init_glfw()  # ?: this will open up the window and let the user wait, should we move this up?
        self.init_imgui()

        args.type = 'gui'
        self.init_opengl()
        self.init_quad()
        self.bind_callbacks()

        self.meshes = []
        self.camera_path = CameraPath()
        self.visualize_axes = True
        self.visualize_paths = True
        self.visualize_cameras = True
        self.visualize_bounds = True
        self.epoch = 0
        self.runner = dotdict(ep_iter=0, collect_timing=False, timer_record_to_file=False, timer_sync_cuda=True)
        self.dataset = dotdict(frame_range=50)
        self.visualization_type = Visualization.RENDER

        # Initialize other parameters
        self.show_demo_window = False
        self.show_metrics_window = False

        # Others
        self.skip_exception = False
        self.static = dotdict(batch=dotdict(), output=dotdict())  # static data store updated through the rendering
        self.dynamic = dotdict()

    def init_camera(self, camera_cfg: dotdict):
        self.camera = Camera(**camera_cfg)
        self.camera.front = self.camera.front  # perform alignment correction

    def render(self):
        global image
        event.wait()
        event.clear()

        with lock:
            buffer = image

        if buffer.shape[0] == self.H and buffer.shape[1] == self.W:
            self.quad.copy_to_texture(buffer)
            self.quad.draw()
        return None, None

    def draw_banner_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):
        imgui.push_font(self.bold_font)
        imgui.text(f'EasyVolcap WebSocket Viewer')
        imgui.text(f'Running on remote: {uri}')
        imgui.pop_font()

    def draw_rendering_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        # Other rendering options like visualization type
        if imgui.collapsing_header('Rendering'):
            self.visualize_axes = imgui_toggle.toggle('Visualize axes', self.visualize_axes, config=self.static.toggle_ios_style)[1]
            self.visualize_bounds = imgui_toggle.toggle('Visualize bounds', self.visualize_bounds, config=self.static.toggle_ios_style)[1]
            self.visualize_cameras = imgui_toggle.toggle('Visualize cameras', self.visualize_cameras, config=self.static.toggle_ios_style)[1]

    def draw_model_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):
        pass

    def draw_model_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):
        pass


async def websocket_client():
    global image
    global viewer
    async with websockets.connect(uri) as websocket:

        while True:
            timer.record('other', log_interval=2.0)

            camera_data = zlib.compress(viewer.camera.to_string().encode('ascii'))
            timer.record('stringify', log_interval=2.0)

            await websocket.send(camera_data)
            timer.record('send', log_interval=2.0)

            buffer = await websocket.recv()
            timer.record('receive', log_interval=2.0)

            try:
                buffer = decode_jpeg(torch.from_numpy(np.frombuffer(buffer, np.uint8)), device='cuda')  # 10ms for 1080p...
            except RuntimeError as e:
                buffer = decode_jpeg(torch.from_numpy(np.frombuffer(buffer, np.uint8)), device='cpu')  # 10ms for 1080p...
            buffer = buffer.permute(1, 2, 0)
            buffer = torch.cat([buffer, torch.ones_like(buffer[..., :1])], dim=-1)

            with lock:
                image = buffer  # might be a cuda tensor or cpu tensor

            event.set()  # explicit synchronization
            timer.record('decode', log_interval=2.0)

uri = "ws://10.76.5.252:1024"
image = None
viewer = Viewer()
lock = threading.Lock()
event = threading.Event()
timer.disabled = False


def start_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(websocket_client())
    loop.run_forever()


client_thread = threading.Thread(target=start_client, daemon=True)
client_thread.start()
catch_throw(viewer.run)()
