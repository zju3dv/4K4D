# For type annotation
from __future__ import annotations

import os
import glm
import sys
import time
import torch
import ctypes
import platform
import subprocess
import numpy as np
from bdb import BdbQuit
from os.path import join
from functools import partial
from collections import deque
from datetime import datetime
from copy import copy, deepcopy
from typing import List, Union, Dict
from glm import vec3, vec4, mat3, mat4, mat4x3
from imgui_bundle import imgui_color_text_edit as ed
from imgui_bundle import portable_file_dialogs as pfd
from imgui_bundle import imgui, imguizmo, imgui_toggle, immvision, implot, ImVec2, ImVec4, imgui_md, immapp, hello_imgui
# Always import glfw *after* imgui_bundle
# (since imgui_bundle will set the correct path where to look for the correct version of the glfw dynamic library)
import glfw

from easyvolcap.engine import cfg  # need this for initialization?
from easyvolcap.models.noop_model import NoopModel
from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner

from easyvolcap.engine import RUNNERS  # controls the optimization loop of a particular epoch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.color_utils import cm_cpu_store
from easyvolcap.utils.image_utils import interpolate_image, resize_image
from easyvolcap.utils.prof_utils import setup_profiler, profiler_step, profiler_start, profiler_stop
from easyvolcap.utils.imgui_utils import push_button_color, pop_button_color, tooltip, colored_wrapped_text
from easyvolcap.utils.viewer_utils import Camera, CameraPath, visualize_cameras, visualize_cube, add_debug_line, add_debug_text, visualize_axes, add_debug_text_2d
from easyvolcap.utils.data_utils import add_batch, add_iter, to_cpu, to_cuda, to_tensor, to_x, default_convert, default_collate, save_image, load_image, Visualization


@RUNNERS.register_module()
class VolumetricVideoViewer:
    # Viewer should be used in conjuction with another runner, which explicitly handles model loading
    def __init__(self,
                 runner: VolumetricVideoRunner,  # already built outside of this init

                 window_size: List[int] = [768, 1366],  # height, width
                 window_title: str = f'EasyVolcap',  # MARK: global config
                 exp_name: str = cfg.exp_name,

                 font_size: int = 18,
                 font_bold: str = 'assets/fonts/CascadiaCodePL-Bold.otf',
                 font_italic: str = 'assets/fonts/CascadiaCodePL-Italic.otf',
                 font_default: str = 'assets/fonts/CascadiaCodePL-Regular.otf',
                 icon_file: str = 'assets/imgs/easyvolcap.png',

                 autoplay_speed: float = 0.005,  # 100 frames for a full loop
                 autoplay: bool = False,
                 visualize_cameras: bool = False,  # will add an extra 5ms on a 3060
                 visualize_bounds: bool = False,  # will add an extra 0.5ms
                 visualize_paths: bool = True,  # might be laggy for larger datasets
                 visualize_axes: bool = False,  # will add an extra 0.xms
                 render_network: bool = True,
                 render_meshes: bool = True,
                 render_alpha: bool = True,

                 update_fps_time: float = 0.1,  # be less stressful
                 update_mem_time: float = 0.1,  # be less stressful
                 use_quad_draw: bool = False,  # different rendering solution
                 use_quad_cuda: bool = True,
                 use_vsync: bool = False,

                 # This is important for works like K-planes or IBR (or stableenerf), since it's not easy to perform interpolation (slow motion)
                 # For point clouds, only a fixed number of point clouds are produces since we performed discrete training (no interpolation)
                 discrete_t: bool = True,  # snap the rendering frame to the closest frame in the dataset
                 playing_fps: int = 30,

                 mesh_preloading: List[str] = [],
                 splat_preloading: List[str] = [],
                 show_preloading: bool = False,
                 skip_exception: bool = False,  # always pause to give user a debugger
                 compose: bool = False,
                 compose_power: float = 1.0,
                 render_ratio: float = 1.0,
                 use_window_focal: bool = False,

                 fullscreen: bool = False,
                 camera_cfg: dotdict = dotdict(type=Camera.__name__),

                 # Debugging elements
                 load_keyframes_at_init: bool = False,
                 show_metrics_window: bool = False,
                 show_demo_window: bool = False,
                 ) -> None:
        # Camera related configurations
        self.camera_cfg = camera_cfg
        self.fullscreen = fullscreen
        self.window_size = window_size
        self.window_title = window_title
        self.use_vsync = use_vsync
        self.use_window_focal = use_window_focal

        # Quad related configurations
        self.use_quad_draw = use_quad_draw
        self.use_quad_cuda = use_quad_cuda
        self.compose = compose  # composing only works with cudagl for now
        self.compose_power = compose_power

        # Font related config
        self.font_default = font_default
        self.font_italic = font_italic
        self.font_bold = font_bold
        self.font_size = font_size
        self.icon_file = icon_file

        # Runner initialization
        self.exp_name = exp_name
        self.runner = runner
        self.runner.visualizer.store_alpha_channel = True  # disable alpha channel for rendering on viewer
        self.runner.visualizer.uncrop_output_images = False  # manual uncropping
        self.epoch = self.runner.load_network()  # load weights only (without optimizer states)
        self.dataset = self.runner.val_dataloader.dataset
        self.model = self.runner.model
        self.model.eval()

        self.init_camera(camera_cfg)  # prepare for the actual rendering now, needs dataset -> needs runner
        self.init_glfw()  # ?: this will open up the window and let the user wait, should we move this up?
        self.init_imgui()
        self.init_opengl()
        self.init_quad()
        self.bind_callbacks()

        # Initialize FPS counter
        self.update_fps_time = update_fps_time
        self.update_mem_time = update_mem_time

        # Initialize animation related stuff
        self.camera_path = CameraPath()
        if load_keyframes_at_init:
            try:
                self.camera_path.load_keyframes(self.dataset.data_root)
            except:
                log(yellow('Unable to load training cameras as keyframes, skipping...'))

        # Initialize temporal controls
        self.playing_speed = autoplay_speed
        self.playing = autoplay

        # Initialize other parameters
        self.show_demo_window = show_demo_window
        self.show_metrics_window = show_metrics_window
        self.visualization_type = Visualization.RENDER
        self.visualize_cameras = visualize_cameras
        self.visualize_bounds = visualize_bounds
        self.visualize_paths = visualize_paths
        self.visualize_axes = visualize_axes

        # Rendering control
        self.exposure = 1.0
        self.offset = 0.0
        self.iter = self.epoch * self.runner.ep_iter  # loaded iter
        from easyvolcap.utils.gl_utils import Mesh, Splat

        self.meshes: List[Mesh] = [
            *[Mesh(filename=mesh, visible=show_preloading) for mesh in mesh_preloading],
            *[Splat(filename=splat, visible=show_preloading, point_radius=0.0015, H=self.H, W=self.W) for splat in splat_preloading],
        ]
        self.render_ratio = render_ratio
        self.render_alpha = render_alpha
        self.render_meshes = render_meshes
        self.render_network = render_network
        self.discrete_t = discrete_t

        # Timinigs
        self.acc_time = 0
        self.prev_time = 0
        self.playing_fps = playing_fps

        # Others
        self.skip_exception = skip_exception
        self.static = dotdict(batch=dotdict(), output=dotdict())  # static data store updated through the rendering
        self.dynamic = dotdict()

    @property
    def render_ratio(self): return self.dataset.render_ratio

    @render_ratio.setter
    def render_ratio(self, v: float): self.dataset.render_ratio = v

    @property
    def window_size(self): return self.H, self.W

    @window_size.setter
    def window_size(self, window_size: List[int]): self.H, self.W = window_size

    @property
    def H(self): return self.camera.H

    @property
    def W(self): return self.camera.W

    @H.setter
    def H(self, v: int): self.camera.H = v

    @W.setter
    def W(self, v: int): self.camera.W = v

    @property
    def camera(self):
        if not hasattr(self, 'render_camera'): self.render_camera = Camera(**self.camera_cfg)  # create the camera object only once
        return self.render_camera

    @camera.setter
    def camera(self, camera: Camera):
        if camera is None: return
        camera = camera.to_batch()  # always operate on batch instead

        H, W = self.window_size  # dimesions
        M = max(H, W)
        Hc, Wc = camera.H, camera.W
        Mc = max(Hc, Wc)
        ratio = M / Mc
        K = camera.K.clone()
        K[:2] *= ratio

        camera.update(dotdict(H=H, W=W, K=K))
        self.render_camera.from_batch(camera)

    def run(self):
        while not glfw.window_should_close(self.window):
            self.frame()
        self.shutdown()

    def frame(self):
        import OpenGL.GL as gl
        # Clear frame buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.dynamic = dotdict()

        # should_render = glfw.get_window_attrib(self.window, glfw.FOCUSED) and self.H > 0 and self.W > 0 and self.camera.H > 0 and self.camera.W > 0

        # Since our framebuffer is not the default framebuffer, the rendering commands will have no impact on the visual output of your window. For this reason it is called off-screen rendering when rendering to a different framebuffer.
        # It is also possible to bind a framebuffer to a read or write target specifically by binding to GL_READ_FRAMEBUFFER or GL_DRAW_FRAMEBUFFER respectively. The framebuffer bound to GL_READ_FRAMEBUFFER is then used for all read operations like glReadPixels and the framebuffer bound to GL_DRAW_FRAMEBUFFER is used as the destination for rendering, clearing and other write operations. Most of the times you won't need to make this distinction though and you generally bind to both with GL_FRAMEBUFFER.

        # Render meshes (or point clouds)
        if self.render_meshes:
            for mesh in self.meshes:
                mesh.render(self.camera)

        # Render network (forward)
        if self.render_network:  # HACK: Will always render the first network stream
            batch, output = self.render()
            self.static.batch = batch
            self.static.output = output

        # Render GUI
        self.draw_imgui(self.static.batch, self.static.output)  # defines GUI elements
        self.show_imgui()

        # Maybe remember profiling results
        profiler_step()

    def render(self):
        # Perform dataloading and forward rendering
        gui_time = timer.record()
        batch = self.camera.to_batch()
        batch = self.dataset.get_viewer_batch(batch)
        batch = add_iter(batch, self.iter, self.runner.total_iter)
        data_time = timer.record()

        batch = to_cuda(add_batch(batch))  # int -> tensor -> add batch -> cuda, smalle operations are much faster on cpu
        ctog_time = timer.record()

        # Forward pass
        self.runner.maybe_jit_model(batch)
        with torch.inference_mode(self.runner.test_using_inference_mode), torch.no_grad(), torch.cuda.amp.autocast(enabled=self.runner.test_use_amp, cache_enabled=self.runner.test_amp_cached):
            try:
                output = self.model(batch)
            except Exception as e:
                if isinstance(e, BdbQuit): raise e
                if self.skip_exception:
                    stacktrace()
                    log(red(f'{type(e)}: {e} encountered when running forward pass, most likely a camera parameter issue, press `R` to to reset camera.'))
                    return batch, dotdict()
                else:
                    raise e
        model_time = timer.record()

        # Filter contents and render to screen
        image = self.runner.visualizer.generate_type(output, batch, self.visualization_type)[0][0]  # RGBA (should we use alpha?)
        if self.exposure != 1.0 or self.offset != 0.0:
            image = torch.cat([(image[..., :3] * self.exposure + self.offset), image[..., -1:]], dim=-1)  # add manual correction
        if 'orig_h' in batch.meta:
            x, y, w, h = batch.meta.crop_x[0].item(), batch.meta.crop_y[0].item(), batch.meta.W[0].item(), batch.meta.H[0].item()
        else:
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        if self.render_ratio != 1.0:
            # NOTE: Principle: resize first (scaling), then crop (translation)
            x, y, w, h = int(x / self.render_ratio), int(y / self.render_ratio), int(w / self.render_ratio), int(h / self.render_ratio)  # FIXME: Never will be an exact match of pixels
            image = resize_image(image, size=(h, w)).contiguous()
        image = (image.clip(0, 1) * 255).type(torch.uint8).flip(0)  # transform
        self.image = image if self.render_alpha else image[..., :3]  # record the image for later use
        post_time = timer.record()

        # The blitting or quad drawing to move texture onto screen
        self.quad.copy_to_texture(image, x, self.H - h - y, w, h)
        self.quad.draw(x, self.H - h - y, w, h)

        gtos_time = timer.record()

        batch.gui_time = gui_time  # gui time of previous frame
        batch.data_time = data_time
        batch.ctog_time = ctog_time
        batch.model_time = model_time
        batch.post_time = post_time
        batch.gtos_time = gtos_time  # previous frame
        return batch, output

    def add_debug_text_2d(viewer, text: str, color: int = 0xff8080ff):
        # slow_pytorch3d_render_msg = 'Using slow PyTorch3D rendering backend. Please see the errors in the terminal.'
        if 'debug_text_2d_loc' not in viewer.dynamic:
            viewer.dynamic.debug_text_2d_loc = ImVec2(0, 0)
        add_debug_text_2d(viewer.dynamic.debug_text_2d_loc, text, color)
        viewer.dynamic.debug_text_2d_loc = ImVec2(0, viewer.dynamic.debug_text_2d_loc.y + 20)

    def draw_cameras(self,
                     Ks: torch.Tensor,
                     Rs: torch.Tensor,
                     Ts: torch.Tensor,
                     proj: mat4,
                     default_color: int = 0xffc08080,
                     bold_color: int = 0xff8080ff,
                     src_inds: torch.Tensor = None):
        # Add to imgui rendering list
        imgui.push_font(self.bold_font)

        # TODO: Slow CPU computation
        for i, (K, R, T) in enumerate(zip(Ks, Rs, Ts)):  # render cameras of this frame

            # Prepare for colors (vscode displays rgba while python uses abgr (little endian -> big endian))
            col = default_color
            thickness = 6.0
            if src_inds is not None and i in src_inds.ravel():
                col = bold_color
                thickness = 12.0

            # Prepare for rendering params
            ixt = mat3(*K.mT.ravel())
            w2c = mat4(mat3(*R.mT.ravel()))
            w2c[3] = vec4(*T.ravel(), 1.0)  # assign translation (not that glm.translate doesn't work)
            c2w = glm.affineInverse(w2c)
            c2w = mat4x3(c2w)
            visualize_cameras(proj, ixt, c2w, col=col, thickness=thickness, name=str(i))
        imgui.pop_font()

    def draw_camera_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        # Camera controls
        self.camera.t = imgui.slider_float('t', self.camera.t, 0, 1, format='%.6f')[1]  # temporal interpolation
        if imgui.collapsing_header('Camera'):
            self.camera.mass = imgui.slider_float('Mass', self.camera.mass, 0.01, 1.0)[1]  # temporal interpolation
            self.camera.moment_of_inertia = imgui.slider_float('Moment of inertia', self.camera.moment_of_inertia, 0.01, 1.0)[1]  # temporal interpolation
            self.camera.movement_force = imgui.slider_float('Movement force', self.camera.movement_force, 1.0, 10.0)[1]  # temporal interpolation
            self.camera.movement_torque = imgui.slider_float('Movement torque', self.camera.movement_torque, 1.0, 10.0)[1]  # temporal interpolation
            self.camera.movement_speed = imgui.slider_float('Mouse speed', self.camera.movement_speed, 0.001, 10.0)[1]  # temporal interpolation
            self.camera.pause_physics = imgui_toggle.toggle('Pause physics', self.camera.pause_physics, config=self.static.toggle_ios_style)[1]
            if imgui.tree_node_ex('Intrinsics'):
                imgui.push_item_width(self.static.slider_width * 0.5)
                changed, value = imgui.slider_float('fx', self.camera.fx, 1.0, self.W * 3, format='%.6f')
                imgui.same_line()  # near bound
                if changed: self.camera.fx = value
                changed, value = imgui.slider_float('fy', self.camera.fx, 1.0, self.H * 3, format='%.6f')
                if changed: self.camera.fy = value
                self.camera.cx = imgui.slider_float('cx', self.camera.cx, 0.0, self.W * 1, format='%.6f')[1]
                imgui.same_line()  # near bound
                self.camera.cy = imgui.slider_float('cy', self.camera.cy, 0.0, self.H * 1, format='%.6f')[1]
                imgui.pop_item_width()
                imgui.tree_pop()

            if imgui.tree_node_ex('Extrinsics'):
                imgui.input_float3('Right', self.camera.right, flags=imgui.InputTextFlags_.read_only, format='%.6f')
                imgui.input_float3('Down', self.camera.down, flags=imgui.InputTextFlags_.read_only, format='%.6f')
                changed, front = imgui.input_float3('Front', self.camera.front, format='%.6f')  # changed, value
                if changed: self.camera.front = vec3(front)
                changed, center = imgui.input_float3('Center', self.camera.center, format='%.6f')
                if changed: self.camera.center = vec3(center)
                imgui.tree_pop()

            if imgui.tree_node_ex('Alignment'):
                self.camera.origin = vec3(imgui.input_float3('Origin', self.camera.origin, format='%.6f')[1])
                self.camera.world_up = vec3(imgui.input_float3('World up', self.camera.world_up, format='%.6f')[1])
                imgui.tree_pop()

            if imgui.tree_node_ex('Bounds & range'):
                if 'log10_size' not in self.static: self.static.log10_size = 1.0  # 10m range
                self.static.log10_size = imgui.slider_float('Bound log10 size (slider range)', self.static.log10_size, -2.0, 5.0)[1]
                size = 10 ** self.static.log10_size
                self.camera.bounds[0] = vec3(imgui.slider_float3('Min x, y, z', self.camera.bounds[0], -size, size, format='%.6f')[1])
                self.camera.bounds[1] = vec3(imgui.slider_float3('Max x, y, z', self.camera.bounds[1], -size, size, format='%.6f')[1])

                imgui.push_item_width(self.static.slider_width * 0.5)
                self.camera.n = imgui.slider_float('Near', self.camera.n, 0.002, self.camera.f - 0.01, format='%.6f')[1]
                imgui.same_line()  # near bound
                self.camera.f = imgui.slider_float('Far', self.camera.f, self.camera.n + 0.01, 100, format='%.6f')[1]  # far bound
                imgui.pop_item_width()
                imgui.tree_pop()

            if imgui.tree_node_ex('Temporal'):
                self.camera.v = imgui.slider_float('v', self.camera.v, 0, 1, format='%.6f')[1]  # spacial interpolation
                self.camera.t = imgui.slider_float('t', self.camera.t, 0, 1, format='%.6f')[1]  # temporal interpolation
                imgui.push_item_width(self.static.slider_width * 0.33)
                self.playing = imgui_toggle.toggle('Autoplay', self.playing, config=self.static.toggle_ios_style)[1]
                imgui.same_line()
                self.discrete_t = imgui_toggle.toggle('Discrete time', self.discrete_t, config=self.static.toggle_ios_style)[1]
                imgui.pop_item_width()
                if self.discrete_t: self.playing_fps = imgui.slider_int('Video FPS', self.playing_fps, 10, 120)[1]  # temporal interpolation
                else: self.playing_speed = imgui.slider_float('Video Speed', self.playing_speed, 0.0001, 0.1)[1]  # temporal interpolation
                imgui.tree_pop()

        # Other updates
        curr_time = time.perf_counter()
        if 'last_time' not in self.static: frame_time = 0
        else: frame_time = curr_time - self.static.last_time
        self.static.last_time = curr_time
        self.camera.step(frame_time)

        # Other updates
        if self.playing:  # automatic update of temporal information
            if self.discrete_t:
                old_prev_time = self.prev_time
                self.prev_time = time.perf_counter()
                self.acc_time += self.prev_time - old_prev_time  # time passed
                frame_time = 1 / self.playing_fps
                if self.acc_time >= frame_time:
                    self.acc_time = 0
                    self.camera.t = (self.camera.t + 1 / self.dataset.frame_range) % 1
            else:
                self.camera.t = (self.camera.t + self.playing_speed) % 1

    def draw_rendering_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        # Other rendering options like visualization type
        if imgui.collapsing_header('Rendering'):
            network_available = not isinstance(self.model, NoopModel)

            if network_available and imgui.begin_combo(f'Visualization', self.visualization_type.name):
                for t in self.runner.visualizer.types:
                    if imgui.selectable(t.name, self.visualization_type == t)[1]:
                        self.visualization_type = t  # construct enum from name
                    if self.visualization_type == t:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            # Maybe implement colormap for other scalars like alpha
            if network_available and imgui.begin_combo(f'Scalar colormap', self.runner.visualizer.dpt_cm):
                for cm in cm_cpu_store:
                    if imgui.selectable(cm, self.runner.visualizer.dpt_cm == cm)[1]:
                        self.runner.visualizer.dpt_cm = cm  # construct enum from name
                    if self.runner.visualizer.dpt_cm == cm:
                        imgui.set_item_default_focus()
                imgui.end_combo()

            self.visualize_axes = imgui_toggle.toggle('Visualize axes', self.visualize_axes, config=self.static.toggle_ios_style)[1]
            self.visualize_bounds = imgui_toggle.toggle('Visualize bounds', self.visualize_bounds, config=self.static.toggle_ios_style)[1]
            self.visualize_cameras = imgui_toggle.toggle('Visualize cameras', self.visualize_cameras, config=self.static.toggle_ios_style)[1]
            if network_available: self.render_network = imgui_toggle.toggle('Render network', self.render_network, config=self.static.toggle_ios_style)[1]
            self.render_meshes = imgui_toggle.toggle('Render meshes', self.render_meshes, config=self.static.toggle_ios_style)[1]
            if network_available: self.render_alpha = imgui_toggle.toggle('Render alpha', self.render_alpha, config=self.static.toggle_ios_style)[1]
            if network_available: self.quad.compose = imgui_toggle.toggle('Compose them', self.quad.compose, config=self.static.toggle_ios_style)[1]
            if network_available: self.quad.use_quad_draw = imgui_toggle.toggle('Drawing quad', self.use_quad_draw, config=self.static.toggle_ios_style)[1]  # 1-2ms faster on wsl
            if network_available:
                self.quad.use_quad_cuda = imgui_toggle.toggle('##cuda_gl_interop', self.quad.use_quad_cuda, config=self.static.toggle_ios_style)[1]
                imgui.same_line()
                imgui.push_font(self.bold_font)
                colored_wrapped_text(0x55cc33ff, 'Use CUDA-GL interop')
                tooltip('If your system does not support CUDA-GL interop (WSL2), please disable this option.')
                imgui.pop_font()

            # Model specific rendering options
            # TODO: Move these to the render_imgui method of 4K4D's samplers
            if hasattr(self.model.sampler, 'volume_rendering'):
                self.model.sampler.volume_rendering = imgui_toggle.toggle('Volume rendering', self.model.sampler.volume_rendering, config=self.static.toggle_ios_style)[1]
            if hasattr(self.model.sampler, 'use_cudagl'):
                self.model.sampler.use_cudagl = imgui_toggle.toggle('Use CUDAGL', self.model.sampler.use_cudagl, config=self.static.toggle_ios_style)[1]
            if hasattr(self.model.sampler, 'use_diffgl'):
                self.model.sampler.use_diffgl = imgui_toggle.toggle('Use DIFFGL', self.model.sampler.use_diffgl, config=self.static.toggle_ios_style)[1]

            # Special care for realtime4dv rendering
            if hasattr(self.model.sampler, 'cudagl'):
                if self.model.sampler.volume_rendering:
                    # Control the shape of the points
                    self.model.sampler.cudagl.point_smooth = imgui_toggle.toggle('Point smooth', self.model.sampler.cudagl.point_smooth, config=self.static.toggle_ios_style)[1]
                    # Control the blending mode of the points
                    self.model.sampler.cudagl.alpha_blending = imgui_toggle.toggle('Alpha blending', self.model.sampler.cudagl.alpha_blending, config=self.static.toggle_ios_style)[1]

            # Handle background color change (and with a random bg color switch)
            if hasattr(self.model.renderer, 'bg_brightness') or hasattr(self.model.sampler, 'bg_brightness'):
                # Check which mode are we in? Custom sampler or full model? The order matters
                if hasattr(self.model.renderer, 'bg_brightness'):
                    bg_brightness = self.model.renderer.bg_brightness
                    def set_bg_brightness(val: float): self.model.renderer.bg_brightness = val  # a type of pointer & reference
                if hasattr(self.model.sampler, 'bg_brightness'):
                    bg_brightness = self.model.sampler.bg_brightness
                    def set_bg_brightness(val: float): self.model.sampler.bg_brightness = val  # a type of pointer & reference

                # Check if we are using random background
                is_random_bkgd = imgui_toggle.toggle('Random bkgd', bg_brightness < 0, config=self.static.toggle_ios_style)[1]

                # Tet the user's choice of background color
                bg_brightness = -1.0 if is_random_bkgd else (bg_brightness if bg_brightness >= 0 else 0.0)  # 1-2ms faster on wsl
                bg_brightness = imgui.slider_float('Bkgd brightness', bg_brightness, 0.0, 1.0)[1]  # not always clamp
                set_bg_brightness(bg_brightness)

                if not is_random_bkgd:
                    # Set bg color for the whole window
                    import OpenGL.GL as gl
                    bg_brightness = np.clip(bg_brightness, 0.0, 1.0)
                    gl.glClearColor(bg_brightness, bg_brightness, bg_brightness, 1.)
            else:
                if 'bg_brightness' not in self.static: self.static.bg_brightness = 0.0
                self.static.bg_brightness = imgui.slider_float('Bkgd brightness', self.static.bg_brightness, 0.0, 1.0)[1]  # not always clamp

                # Set bg color for the whole window
                import OpenGL.GL as gl
                bg_brightness = np.clip(self.static.bg_brightness, 0.0, 1.0)
                gl.glClearColor(bg_brightness, bg_brightness, bg_brightness, 1.)

            # Almost all models has this option
            if network_available: self.quad.compose_power = imgui.slider_float('Compose power', self.quad.compose_power, 1.0, 10.0)[1]  # temporal interpolation
            if network_available: self.exposure = imgui.slider_float('Exposure', self.exposure, 0.0, 100.0)[1]  # temporal interpolation
            if network_available: self.offset = imgui.slider_float('Offset', self.offset, 0.0, 1.0)[1]  # temporal interpolation

            if network_available: self.render_ratio = imgui.slider_float('Render ratio', self.render_ratio, 0.01, 2.0, flags=imgui.SliderFlags_.always_clamp)[1]  # temporal interpolation
            if network_available: self.model.render_chunk_size = imgui.slider_int('Render chunk size', self.model.render_chunk_size, 512, 1048576)[1]

            # Special care for realtime4dv rendering
            if hasattr(self.model.sampler, 'cudagl'):
                # Control the radius of the points
                if self.model.sampler.volume_rendering: value = self.model.sampler.cudagl.radii_mult_volume
                else: value = self.model.sampler.cudagl.radii_mult_solid
                value = imgui.slider_float('Splatting radii_mult', value, 0.01, 3.0)[1]
                if self.model.sampler.volume_rendering: self.model.sampler.cudagl.radii_mult_volume = value
                else: self.model.sampler.cudagl.radii_mult_solid = value

            # Special care for realtime4dv rendering
            if hasattr(self.model.sampler, 'pts_per_pix'):
                self.model.sampler.pts_per_pix = imgui.slider_int('Splatting pts_per_pix', self.model.sampler.pts_per_pix, 1, 60)[1]

        # Custome GUI elements will be rendered here
        if hasattr(self.model, 'render_imgui'):
            self.model.render_imgui(self, batch)  # some times the model has its own GUI elements

        if not self.quad.use_quad_cuda:
            gpu_cpu_gpu_msg = f'Not using CUDA-GL interop for low-latency upload, will lead to degraded performance. Try using native Windows or Linux for CUDA-GL interop'

            imgui.push_font(self.bold_font)
            colored_wrapped_text(0xff3355ff, gpu_cpu_gpu_msg)
            self.add_debug_text_2d(gpu_cpu_gpu_msg, 0xff3355ff)
            imgui.pop_font()

        # Render debug cameras out (this should not be affected bu guis)
        if self.visualize_axes or self.visualize_cameras or self.visualize_bounds or self.camera_path.keyframes:
            proj = self.camera.w2p  # 3, 4

        if self.visualize_cameras and hasattr(self.dataset, 'Ks'):
            # Prepare tensors to render
            dataset = self.dataset
            if hasattr(dataset, 'closest_using_t') and dataset.closest_using_t:
                Ks = dataset.Ks[batch.meta.view_index.item()]  # avoid erronous selections
                Rs = dataset.Rs[batch.meta.view_index.item()]
                Ts = dataset.Ts[batch.meta.view_index.item()]
            else:
                Ks = dataset.Ks[:, batch.meta.latent_index.item()]
                Rs = dataset.Rs[:, batch.meta.latent_index.item()]
                Ts = dataset.Ts[:, batch.meta.latent_index.item()]

            if 'src_inds' in batch.meta: src_inds = batch.meta.src_inds
            else: src_inds = None
            self.draw_cameras(Ks, Rs, Ts, proj, src_inds=src_inds)

            if 'optimized_camera' in self.static:
                if hasattr(dataset, 'closest_using_t') and dataset.closest_using_t:
                    Ks = self.static.optimized_camera.Ks[batch.meta.view_index.item()]  # avoid erronous selections
                    Rs = self.static.optimized_camera.Rs[batch.meta.view_index.item()]
                    Ts = self.static.optimized_camera.Ts[batch.meta.view_index.item()]
                else:
                    Ks = self.static.optimized_camera.Ks[:, batch.meta.latent_index.item()]
                    Rs = self.static.optimized_camera.Rs[:, batch.meta.latent_index.item()]
                    Ts = self.static.optimized_camera.Ts[:, batch.meta.latent_index.item()]

                self.draw_cameras(Ks, Rs, Ts, proj, default_color=0xeeffc050, bold_color=0xffffa020, src_inds=src_inds)

            elif hasattr(self.model.camera, 'pose_resd'):
                meta = dotdict()
                meta.K = dataset.Ks.view(np.prod(dataset.Ks.shape[:2]), *dataset.Ks.shape[2:])
                meta.R = dataset.Rs.view(np.prod(dataset.Rs.shape[:2]), *dataset.Rs.shape[2:])
                meta.T = dataset.Ts.view(np.prod(dataset.Ts.shape[:2]), *dataset.Ts.shape[2:])
                latent_index = torch.arange(dataset.n_latents)
                view_index = torch.arange(dataset.n_views)
                meta.latent_index, meta.view_index = torch.meshgrid(latent_index, view_index, indexing='ij')
                meta.latent_index = meta.latent_index.reshape(dataset.n_latents * dataset.n_views)
                meta.view_index = meta.view_index.reshape(dataset.n_latents * dataset.n_views)

                fake_batch = to_cuda(meta)
                fake_batch = self.model.prepare_camera(fake_batch)
                fake_batch.meta = meta
                with torch.no_grad() and torch.inference_mode():
                    fake_batch = self.model.camera.forward_cams(fake_batch)
                self.static.optimized_camera = dotdict()
                self.static.optimized_camera.Ks = fake_batch.K.view(dataset.Ks.shape).float()
                self.static.optimized_camera.Rs = fake_batch.R.view(dataset.Rs.shape).float()
                self.static.optimized_camera.Ts = fake_batch.T.view(dataset.Ts.shape).float()

        # Render debug bounding box out
        if self.visualize_bounds:
            bounds = self.camera.bounds if 'bounds' not in batch.meta else batch.meta.bounds[0]
            visualize_cube(proj, vec3(*bounds[0]), vec3(*bounds[1]), thickness=6.0)  # bounding box

        if self.visualize_axes:
            visualize_axes(proj, vec3(0, 0, 0), vec3(0.1, 0.1, 0.1), thickness=6.0, name='World')  # bounding box
            visualize_axes(proj, self.camera.origin, self.camera.origin + vec3(0.1, 0.1, 0.1), thickness=6.0, name='Rotation')  # bounding box

    def draw_model_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        if imgui.collapsing_header(f'Model & network'):
            if imgui.button('Reload model'):
                try:
                    self.epoch = self.runner.load_network()
                    if 'optimized_camera' in self.static: del self.static.optimized_camera
                    self.iter = self.epoch * self.runner.ep_iter  # loaded iter
                except: pass  # sometimes the training process is still saving data

            imgui.same_line()
            push_button_color(0xff3355ff)
            if (imgui.button('Reset viewer')): self.reset()
            pop_button_color()

            imgui.text(f'Current epoch: {self.epoch}')
            self.iter = imgui.slider_int('Iteration override', self.iter, 0, self.runner.total_iter)[1]  # temporal interpolation

    def draw_keyframes_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        # Export animation (camera paths)
        # imgui.set_next_item_open(True)
        if imgui.collapsing_header(f'Animation (keyframes: {len(self.camera_path)})###animation', ):

            push_button_color(0x55cc33ff)
            if imgui.button('Insert'):
                self.camera_path.insert(self.camera)
            pop_button_color()

            if len(self.camera_path):  # if exists, can delete or replace
                # Update the keyframes
                imgui.same_line()
                push_button_color(0xff5533ff)
                if imgui.button('Replace'): self.camera_path.replace(self.camera)
                pop_button_color()

                # Update the keyframes
                imgui.same_line()
                push_button_color(0xff3355ff)
                if imgui.button('Delete'): self.camera_path.delete(self.camera_path.selected)
                pop_button_color()

                # Update the keyframes
                imgui.same_line()
                push_button_color(0xff3355ff)
                if imgui.button('Clear'): self.camera_path.clear()
                pop_button_color()

            # push_button_color(0xff5533ff)
            if imgui.button('Load'):
                self.static.load_keyframes_dialog = pfd.select_folder("Select folder")
            # pop_button_color()
            if 'load_keyframes_dialog' in self.static and \
                    self.static.load_keyframes_dialog is not None and \
                    self.static.load_keyframes_dialog.ready(timeout=1):  # this is not moved up since it spans frames # MARK: SLOW
                directory = self.static.load_keyframes_dialog.result()
                if directory:
                    self.camera_path.load_keyframes(directory)
                    self.static.keyframes_path = directory
                self.static.load_keyframes_dialog = None

            # Timelines
            if len(self.camera_path):  # need at least 3 components to interpolate
                imgui.same_line()
                if imgui.button('Export keyframes'):
                    self.static.export_keyframes_dialog = pfd.select_folder("Select folder")
                if 'export_keyframes_dialog' in self.static and \
                        self.static.export_keyframes_dialog is not None and \
                        self.static.export_keyframes_dialog.ready(timeout=1):  # this is not moved up since it spans frames # MARK: SLOW
                    directory = self.static.export_keyframes_dialog.result()
                    if directory:
                        self.camera_path.export_keyframes(directory)
                        self.static.keyframes_path = directory
                    self.static.export_keyframes_dialog = None

                imgui.same_line()
                if imgui.button('Export interpolated'):
                    self.static.export_interp_dialog = pfd.select_folder("Select folder")
                if 'export_interp_dialog' in self.static and \
                        self.static.export_interp_dialog is not None and \
                        self.static.export_interp_dialog.ready(timeout=1):  # this is not moved up since it spans frames # MARK: SLOW
                    directory = self.static.export_interp_dialog.result()
                    if directory:
                        self.camera_path.export_interps(directory)
                        self.static.keyframes_path = directory
                    self.static.export_interp_dialog = None

                self.camera_path.n_render_views = imgui.slider_int('N Interps', self.camera_path.n_render_views, 100, 10000)[1]  # temporal interpolation

            if len(self.camera_path):  # if exists, can delete or replace
                imgui.text('Timeline control')
                space = (len(self.camera_path) - 1) / len(self.camera_path)  # to fill them up
                width = self.static.slider_width / len(self.camera_path) - space
                for i in range(len(self.camera_path)):
                    if i != 0:
                        imgui.same_line(0, 1)
                    sel = i == self.camera_path.selected  # might get updated during this
                    if sel:
                        push_button_color(0x8855aaff)  #
                    if imgui.button(f'###{i}', ImVec2(width, 0)):
                        self.camera_path.selected = i  # will not change playing_time after inserting the first keyframe
                        self.static.playing_time = self.camera_path.playing_time  # Do not change playing time, instead load the stored camera, this variable controls wherther to interp
                        self.camera = deepcopy(self.camera_path.keyframes[i])  # change the current camera
                    if sel:
                        pop_button_color()

            # Timelines
            if len(self.camera_path) > 3:  # need at least 3 components to interpolate

                # Player control
                imgui.text('Player control')
                if imgui.button(f'{"|<"}'):  # centered
                    self.camera_path.selected = 0
                imgui.same_line()
                if imgui.button(f'{"<"}'):
                    self.camera_path.selected = max(0, self.camera_path.selected - 1)

                imgui.same_line()
                push_button_color(0xff5533ff if self.camera_path.playing else 0x55cc33ff)
                if imgui.button(f'{"Stop": ^4}' if self.camera_path.playing else f'{"Play": ^4}'):
                    self.camera_path.playing = not self.camera_path.playing
                pop_button_color()

                imgui.same_line()
                if imgui.button(f'{">"}'):
                    self.camera_path.selected = min(len(self.camera_path) - 1, self.camera_path.selected + 1)

                imgui.same_line()
                if imgui.button(f'{">|"}'):
                    self.camera_path.selected = len(self.camera_path) - 1

                # if self.camera_path.playing:
                imgui.same_line()
                self.camera_path.playing_speed = imgui.slider_float('Speed', self.camera_path.playing_speed, 0.0001, 0.005, format='%.6f')[1]  # temporal interpolation

                # Timeline slider
                self.camera_path.playing_time = imgui.slider_float('Playing time', self.camera_path.playing_time, 0, 1)[1]  # temporal interpolation
                self.camera_path.loop_interp = imgui_toggle.toggle('Loop interpolations', self.camera_path.loop_interp, config=self.static.toggle_ios_style)[1]
                self.visualize_paths = imgui_toggle.toggle('Visualize paths', self.visualize_paths, config=self.static.toggle_ios_style)[1]

                if 'keyframes_path' in self.static:
                    offline_title = 'Keyframes offline rendering script:'
                    imgui.text(offline_title)
                    backslash, slash = '\\', '/'  # windows supports both forward and backward slash
                    args = sys.argv
                    args[0] = os.path.basename(args[0])  # hope this can be called at whereever place
                    source = f"{' '.join(args)}".replace('gui', 'test')
                    source += " " + f"val_dataloader_cfg.dataset_cfg.camera_path_intri={join(self.static.keyframes_path, 'intri.yml').replace(backslash, slash)}"
                    source += " " + f"val_dataloader_cfg.dataset_cfg.camera_path_extri={join(self.static.keyframes_path, 'extri.yml').replace(backslash, slash)}"
                    source += " " + f"val_dataloader_cfg.dataset_cfg.temporal_range=None"
                    source += " " + f"configs=configs/specs/cubic.yaml,configs/specs/ibr.yaml,configs/specs/cubic.yaml,configs/specs/interp.yaml" if 'ImageBased' in self.dataset.__class__.__name__ else " " + f"configs=configs/specs/interp.yaml"
                    source += " " + f"val_dataloader_cfg.dataset_cfg.interp_cfg.smoothing_term=0.0" if self.camera_path.loop_interp else " " + f"val_dataloader_cfg.dataset_cfg.interp_cfg.smoothing_term=10.0"

                    if 'editor' not in self.static:
                        editor = ed.TextEditor()
                        editor.set_language_definition(ed.TextEditor.LanguageDefinition.python())
                        editor.set_read_only(True)
                        editor.set_show_whitespaces(True)
                        self.static.editor = editor

                    line_height = imgui.get_font_size()
                    editor_size = ImVec2()
                    editor_size.x = (imgui.get_content_region_max().x - imgui.get_window_content_region_min().x - imgui.get_style().item_spacing.x)
                    editor_size.y = line_height * (len(source.split('\n')) + 1.5)

                    if (imgui.button('Copy')):
                        imgui.set_clipboard_text(source)

                    imgui.same_line()
                    self.static.editor.set_text(source)
                    self.static.editor.render(a_title='Code', a_size=editor_size, a_border=False)  # id, size, border

                    if 'offline' in self.static and self.static.offline is not None:  # exists and started
                        if self.static.offline.poll() is None:
                            # Still running
                            push_button_color(0xff3355ff)
                            if (imgui.button('Kill offline rendering')):
                                self.static.offline.kill()
                            pop_button_color()
                            imgui.text(f'Offline rendering running... (PID: {self.static.offline.pid})')
                            imgui.text(f'Please check the terminal output for details')
                        else:
                            # Finished
                            self.static.offline = None
                    else:
                        if (imgui.button('Run offline rendering')):
                            self.static.offline = subprocess.Popen(source.split(' '))

        if self.camera_path.playing:  # automatic update of playing time
            self.camera_path.playing_time = (self.camera_path.playing_time + self.camera_path.playing_speed) % 1

        if self.camera_path.playing_time != self.static.playing_time and len(self.camera_path) > 3:  # ok to interpolate
            # Update main camera
            us = self.camera_path.playing_time
            interp = self.camera_path.interp(us)
            if interp is not None:
                H, W, K, R, T, n, f, t, v, bounds = interp
                interp = dotdict(H=H, W=W, K=K, R=R, T=T, n=n, f=f, t=t, v=v, bounds=bounds)
                self.camera.from_batch(interp)  # may return None

            # Update cursor when dragging the slider
            K = len(self.camera_path)
            self.camera_path.cursor_index = min(int(np.floor(us * (K - 1))), K - 1)  # do not interp or change playtime

        # Render user added camera path
        if self.visualize_paths:
            self.camera_path.draw(self.camera)  # do the projection

    def draw_mesh_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):
        from easyvolcap.utils.gl_utils import Mesh, Splat, Gaussian, PointSplat

        if imgui.collapsing_header('Meshes & splats'):
            if imgui.button('Add triangle mesh from file'):
                self.static.add_mesh_dialog = pfd.open_file('Select file', filters=['PLY Files', '*.ply'])
                self.static.render_type = Mesh.RenderType.TRIS
                self.static.mesh_class = Mesh
            if imgui.button('Add point cloud from file'):
                self.static.add_mesh_dialog = pfd.open_file('Select file', filters=['PLY Files', '*.ply'])
                self.static.render_type = Mesh.RenderType.POINTS
                self.static.mesh_class = Mesh
            if imgui.button('Add point splat from file'):
                self.static.add_mesh_dialog = pfd.open_file('Select file', filters=['PLY Files', '*.ply'])
                self.static.mesh_class = Splat
            if imgui.button('Add gaussian splat from file'):
                self.static.add_mesh_dialog = pfd.open_file('Select file', filters=['3DGS Files', '*.ply *.npz *.pt *.pth'])
                self.static.mesh_class = Gaussian
            if imgui.button('Add camera path visualization'):
                self.static.add_mesh_dialog = pfd.select_folder('Select folder')
                self.static.mesh_class = CameraPath
            if 'add_mesh_dialog' in self.static and \
               self.static.add_mesh_dialog is not None and \
               self.static.add_mesh_dialog.ready(timeout=1):  # this is not moved up since it spans frames # MARK: SLOW

                directory = self.static.add_mesh_dialog.result()
                if directory:
                    # Prepare arguments for mesh creation
                    kwargs = dotdict()
                    if 'render_type' in self.static: kwargs.render_type = self.static.render_type
                    if 'vert_sizes' in self.static: kwargs.vert_sizes = self.static.vert_sizes

                    # Construct and store the mesh
                    filename = directory[0] if isinstance(directory, list) else directory
                    mesh = self.static.mesh_class(filename=filename, H=self.H, W=self.W, **kwargs)
                    self.meshes.append(mesh)
                self.static.add_mesh_dialog = None

            will_delete = []
            for i, mesh in enumerate(self.meshes):
                mesh.render_imgui(viewer=self, batch=dotdict(i=i, will_delete=will_delete, slider_width=self.static.slider_width))

            for i in will_delete:
                del self.meshes[i]

    def draw_debug_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        if imgui.collapsing_header('Debugging'):
            imgui.text('The UI will freeze to switch control to pdbr')
            imgui.text('Use "c" in pdbr to continue normal execution')
            push_button_color(0xff3355ff)
            if imgui.button('Invoke pdbr (go see the terminal)'):
                breakpoint()  # preserve tqdm (do not use debugger())
            if imgui.button('Enable breakpoint (if defined in code)'):
                enable_breakpoint()  # preserve tqdm (do not use debugger())
            pop_button_color()

    def draw_banner_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        # Misc information
        imgui.push_font(self.bold_font)
        imgui.text(f'EasyVolcap Framework -- by zju3dv')
        imgui.text(f'Running on {self.static.name}: {self.static.device}')
        imgui.text(f'FPS       : {self.static.fps:7.3f} FPS')
        tooltip('This is the 20 frame average fps instead of per frame')
        imgui.text(f'torch VRAM: {self.static.memory / 2**20:7.2f} MB ')
        tooltip('For now, we only show VRAM used by the torch engine. OpenGL VRAM is not counted')
        imgui.text(f'frame time: {self.static.frame_time * 1000:7.3f} ms ')
        tooltip('This is the 20 frame average frame_time instead of per frame')
        imgui.pop_font()

        # Full frame timings
        self.runner.collect_timing = imgui_toggle.toggle('Collect timing', self.runner.collect_timing, config=self.static.toggle_ios_style)[1]
        changed, value = imgui_toggle.toggle('Record timing', self.runner.timer_record_to_file, config=self.static.toggle_ios_style)
        if changed:
            self.runner.timer_record_to_file = value
        self.runner.timer_sync_cuda = imgui_toggle.toggle('Sync timing', self.runner.timer_sync_cuda, config=self.static.toggle_ios_style)[1]
        changed, self.use_vsync = imgui_toggle.toggle('Enable VSync', self.use_vsync, config=self.static.toggle_ios_style)
        if changed:
            glfw.swap_interval(self.use_vsync)

        if not timer.disabled:
            if imgui.collapsing_header('Timing'):
                imgui.text(f'gui  : {batch.gui_time * 1000:7.3f}ms')
                tooltip('Time for every other GUI elements (imgui & meshes & cameras & bounds) of previous frame')
                imgui.text(f'data : {batch.data_time * 1000:7.3f}ms')
                tooltip('Time for extracting data from GUI & pass through dataset wrapper')
                imgui.text(f'ctog : {batch.ctog_time * 1000:7.3f}ms')
                tooltip('Time for move batch data from CPU RAM to GPU VRAM (limited by PCIe bandwidth)')
                imgui.text(f'model: {batch.model_time * 1000:7.3f}ms')
                tooltip('Time for forwarding the underlying model, this should dominate')
                imgui.text(f'post : {batch.post_time * 1000:7.3f}ms')
                tooltip('Time for post processing of the rendered image got from the model')
                imgui.text(f'gtos : {batch.gtos_time * 1000:7.3f}ms')
                tooltip('Time for blitting the rendered content from torch.Tensor to screen of previous frame')

    def draw_menu_gui(self, batch: dotdict = dotdict(), output: dotdict = dotdict()):

        if imgui.begin_menu_bar():
            if imgui.begin_menu('File', True):
                imgui.end_menu()

            if imgui.begin_menu('Edit', True):
                if imgui.menu_item('Quit Viewer', 'Ctrl+Q', False, True)[0]:
                    exit(1)

                if imgui.menu_item('Toggle Fullscreen', 'F11', False, True)[0]:
                    self.toggle_fullscreen()

                if imgui.menu_item('Toggle Help Window', 'F1', False, True)[0]:
                    pass  # TODO: finish help window

                if imgui.menu_item('Toggle ImGUI Metrics', 'F10', False, True)[0]:
                    self.show_metrics_window = not self.show_metrics_window

                if imgui.menu_item('Toggle ImGUI Demo', 'F12', False, True)[0]:
                    self.show_demo_window = not self.show_demo_window

                imgui.end_menu()
            imgui.end_menu_bar()

        if self.show_metrics_window:
            imgui.show_metrics_window()

        if self.show_demo_window:
            imgui.show_demo_window()

    def draw_imgui(self, batch: dotdict, output: dotdict):  # need to explicitly handle empty input
        # Initialization
        glfw.poll_events()  # process pending events, keyboard and stuff
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()
        imgui.push_font(self.default_font)

        # States
        self.static.playing_time = self.camera_path.playing_time  # Remember this, if changed, update camera
        self.static.slider_width = imgui.get_window_width() * 0.65  # https://github.com/ocornut/imgui/issues/267
        self.static.toggle_ios_style = imgui_toggle.ios_style(size_scale=0.2, light_mode=False)

        # Titles
        fps, frame_time = self.get_fps_and_frame_time()
        name, device, memory = self.get_device_and_memory()
        # glfw.set_window_title(self.window, self.window_title.format(FPS=fps)) # might confuse window managers
        self.static.fps = fps
        self.static.frame_time = frame_time
        self.static.name = name
        self.static.device = device
        self.static.memory = memory

        # Being the main window
        imgui.begin(f'{self.W}x{self.H} {fps:.3f} fps###main', flags=imgui.WindowFlags_.menu_bar)

        self.draw_menu_gui(batch, output)
        self.draw_banner_gui(batch, output)
        self.draw_camera_gui(batch, output)
        self.draw_rendering_gui(batch, output)
        self.draw_keyframes_gui(batch, output)
        self.draw_model_gui(batch, output)
        self.draw_mesh_gui(batch, output)
        self.draw_debug_gui(batch, output)

        # End of gui and rendering
        imgui.end()
        imgui.pop_font()
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def show_imgui(self):
        # Update and Render additional Platform Windows
        # (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        #  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        io = imgui.get_io()
        if io.config_flags & imgui.ConfigFlags_.viewports_enable > 0:
            backup_current_context = glfw.get_current_context()
            imgui.update_platform_windows()
            imgui.render_platform_windows_default()
            glfw.make_context_current(backup_current_context)
        glfw.swap_buffers(self.window)

    def printscreen(self):
        # Open a screen saving dialog and save the content to disk?
        # Or just save the current frame to the current working directory and print a log
        cwd = os.getcwd()
        # imgui.text(f'Saving current frame to {cwd}')
        path = join(cwd, f'{self.exp_name}_{int(datetime.now().timestamp())}.png')
        log(f'Saving screenshot to {blue(path)}')
        save_image(path, self.image.flip(0).detach().cpu().numpy())  # MARK: SYNC

    def glfw_char_callback(self, window, codepoint):
        if (imgui.get_io().want_text_input):
            return imgui.backends.glfw_char_callback(self.window_address, codepoint)

    def glfw_key_callback(self, window, key, scancode, action, mods):
        if (imgui.get_io().want_capture_keyboard):
            return imgui.backends.glfw_key_callback(self.window_address, key, scancode, action, mods)

        CONTROL = mods & glfw.MOD_CONTROL
        SHIFT = mods & glfw.MOD_SHIFT
        ALT = mods & glfw.MOD_ALT

        if (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_Q and CONTROL:
            self.shutdown()

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_S and CONTROL:
            self.printscreen()

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_C and CONTROL:
            import pyperclip
            content = self.camera.to_string()
            pyperclip.copy(content)
            log(yellow(f'Copied camera to clipboard:\n{content}'))

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_V and CONTROL:
            import pyperclip
            content = pyperclip.paste()
            try:
                self.camera.from_string(content)
                log(yellow(f'Pasted camera from clipboard:\n{content}'))
            except:
                log(red(f'Failed to parse the clipboard content: {content}'))

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_LEFT and CONTROL:
            if SHIFT: diff = 10
            else: diff = 1
            self.snap_camera_index = (self.snap_camera_index - diff) % self.dataset.n_views
            log(yellow(f'Snapping to input camera {self.snap_camera_index}'))
            self.init_camera(self.camera_cfg, self.snap_camera_index)

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_RIGHT and CONTROL:
            if SHIFT: diff = 10
            else: diff = 1
            self.snap_camera_index = (self.snap_camera_index + diff) % self.dataset.n_views
            log(yellow(f'Snapping to input camera {self.snap_camera_index}'))
            self.init_camera(self.camera_cfg, self.snap_camera_index)

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_F12:
            self.show_demo_window = not self.show_demo_window  # will render test window

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_F11:
            self.toggle_fullscreen()

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_F10:
            self.show_metrics_window = not self.show_metrics_window  # will render test window

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_SPACE:
            if SHIFT: self.camera_path.playing = not self.camera_path.playing
            else: self.playing = not self.playing  # play automatically

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_N:
            self.render_network = not self.render_network

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_M:
            self.render_meshes = not self.render_meshes

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_R:
            # Reset gui (mainly camera)
            self.reset()

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_I:
            # Insert animation keyframe
            self.camera_path.insert(self.camera)

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_A:
            self.camera.force.x = -self.camera.movement_force

        elif action == glfw.RELEASE and key == glfw.KEY_A:
            self.camera.force.x = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_W:
            self.camera.force.y = self.camera.movement_force

        elif action == glfw.RELEASE and key == glfw.KEY_W:
            self.camera.force.y = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_S:
            self.camera.force.y = -self.camera.movement_force

        elif action == glfw.RELEASE and key == glfw.KEY_S:
            self.camera.force.y = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_D:
            self.camera.force.x = self.camera.movement_force

        elif action == glfw.RELEASE and key == glfw.KEY_D:
            self.camera.force.x = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_C:
            self.camera.force.z = -self.camera.movement_force

        elif action == glfw.RELEASE and key == glfw.KEY_C:
            self.camera.force.z = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_V:
            self.camera.force.z = self.camera.movement_force

        elif action == glfw.RELEASE and key == glfw.KEY_V:
            self.camera.force.z = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_1:
            self.camera.torque.x = -self.camera.movement_torque

        elif action == glfw.RELEASE and key == glfw.KEY_1:
            self.camera.torque.x = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_2:
            self.camera.torque.x = self.camera.movement_torque

        elif action == glfw.RELEASE and key == glfw.KEY_2:
            self.camera.torque.x = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_Q:
            self.camera.torque.y = -self.camera.movement_torque

        elif action == glfw.RELEASE and key == glfw.KEY_Q:
            self.camera.torque.y = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_E:
            self.camera.torque.y = self.camera.movement_torque

        elif action == glfw.RELEASE and key == glfw.KEY_E:
            self.camera.torque.y = 0.0

        elif (action == glfw.PRESS or action == glfw.REPEAT) and key == glfw.KEY_L:
            # Toggle camera path looping
            self.camera_path.loop_interp = not self.camera_path.loop_interp

        elif ((action == glfw.PRESS or action == glfw.REPEAT) or action == glfw.REPEAT) and key == glfw.KEY_LEFT:
            # Backup one frame
            self.camera.t = np.clip(self.camera.t - 1 / self.dataset.frame_range, 0, 1)

        elif ((action == glfw.PRESS or action == glfw.REPEAT) or action == glfw.REPEAT) and key == glfw.KEY_RIGHT:
            # Advance one frame
            self.camera.t = np.clip(self.camera.t + 1 / self.dataset.frame_range, 0, 1)

        elif ((action == glfw.PRESS or action == glfw.REPEAT) or action == glfw.REPEAT) and key >= glfw.KEY_0 and key <= glfw.KEY_9 and CONTROL:
            # Snap to input camera
            view_index = key - glfw.KEY_0
            if len(self.dataset.Rs) - 1 >= view_index:
                log(yellow(f'Snapping to input camera {view_index}'))
                self.init_camera(self.camera_cfg, view_index=view_index)
            else:
                log(yellow(f'Provided view index ({view_index}) is out of range for the dataset ({len(self.dataset.Rs)})'))

    def toggle_fullscreen(self):
        if not self.fullscreen:
            # Get the primary monitor
            self.monitor = glfw.get_primary_monitor()
            self.mode = glfw.get_video_mode(self.monitor)

            # Store the current window size and position
            self.windowed_size = glfw.get_window_size(self.window)
            self.windowed_position = glfw.get_window_pos(self.window)

            # Set the window to full screen
            glfw.set_window_monitor(self.window, self.monitor, 0, 0, self.mode.size.width, self.mode.size.height, self.mode.refresh_rate)
            self.fullscreen = True
        else:
            # Restore windowed mode with original size and position
            glfw.set_window_monitor(self.window, None, self.windowed_position[0], self.windowed_position[1], self.windowed_size[0], self.windowed_size[1], 0)
            self.fullscreen = False

    def get_fps_and_frame_time(self):
        first_run = 'last_fps_update' not in self.static
        curr_time = time.perf_counter()
        if first_run:
            self.static.last_fps_update = curr_time
            self.static.frame_time = 1
            self.static.fps = 1
            self.static.acc_frame = 1
        elif curr_time - self.static.last_fps_update > self.update_fps_time:
            self.static.frame_time = (curr_time - self.static.last_fps_update) / self.static.acc_frame
            self.static.fps = 1 / self.static.frame_time  # in fps
            self.static.last_fps_update = curr_time
            self.static.acc_frame = 1
        else:
            self.static.acc_frame += 1
        return self.static.fps, self.static.frame_time

    def get_device_and_memory(self):
        first_run = 'last_memory_update' not in self.static
        curr_time = time.perf_counter()
        if first_run or curr_time - self.static.last_memory_update > self.update_mem_time:
            self.static.name = torch.cuda.get_device_name()
            self.static.device = next(self.model.parameters()).device
            self.static.memory = torch.cuda.max_memory_allocated()
            self.static.last_memory_update = curr_time
        return self.static.name, self.static.device, self.static.memory

    def shutdown(self):
        # Cleanup
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()

        glfw.destroy_window(self.window)
        glfw.terminate()

    def reset(self):
        self.init_camera(self.camera_cfg)
        self.playing = False
        self.exposure = 1.0
        self.offset = 0.0
        # self.autoplay_speed = 0.01
        self.iter = self.epoch * self.runner.ep_iter  # loaded iter

    def glfw_mouse_button_callback(self, window, button, action, mods):
        # Let the UI handle its corrsponding operations
        if (imgui.get_io().want_capture_mouse):
            imgui.backends.glfw_mouse_button_callback(self.window_address, button, action, mods)
            if action != glfw.RELEASE:
                return  # only return if not releasing the mouse

        x, y = glfw.get_cursor_pos(window)
        if (action == glfw.PRESS or action == glfw.REPEAT):
            SHIFT = mods & glfw.MOD_SHIFT
            CONTROL = mods & glfw.MOD_CONTROL
            MIDDLE = button == glfw.MOUSE_BUTTON_MIDDLE
            LEFT = button == glfw.MOUSE_BUTTON_LEFT
            RIGHT = button == glfw.MOUSE_BUTTON_RIGHT

            is_panning = SHIFT or MIDDLE
            about_origin = LEFT or (MIDDLE and SHIFT)
            self.camera.begin_dragging(x, y, is_panning, about_origin)
        elif action == glfw.RELEASE:
            self.camera.end_dragging()
        else:
            log(red('Mouse button callback falling through'), button)

    def glfw_cursor_pos_callback(self, window, x, y):
        self.camera.update_dragging(x, y)
        return imgui.backends.glfw_cursor_pos_callback(self.window_address, x, y)

    def glfw_scroll_callback(self, window, x_offset, y_offset):
        if (imgui.get_io().want_capture_mouse):
            return imgui.backends.glfw_scroll_callback(self.window_address, x_offset, y_offset)
        CONTROL = glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
        SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        ALT = glfw.get_key(window, glfw.KEY_LEFT_ALT) == glfw.PRESS

        if CONTROL and SHIFT:
            self.camera.move(x_offset, y_offset)
        elif CONTROL:
            self.camera.n = max(self.camera.n + y_offset * 0.01, 0.001)
        elif SHIFT:
            self.camera.f = self.camera.f + y_offset * 0.1
        elif ALT:
            self.render_ratio = self.render_ratio + y_offset * 0.01
            self.render_ratio = min(max(self.render_ratio, 0.05), 1.0)
        else:
            self.camera.fx = self.camera.fx + y_offset * 50  # locked xy

    def glfw_framebuffer_size_callback(self, window, width, height):
        import OpenGL.GL as gl
        ori_H = self.H
        ori_W = self.W
        self.H = height
        self.W = width

        if ori_W > 0 and ori_H > 0:
            ratio = min(self.W / ori_W, self.H / ori_H)
            self.camera.K[0, 0] *= ratio   # only update one of them
            self.camera.K[1, 1] *= ratio   # only update one of them
            self.camera.K[2, 0] *= self.W / ori_W   # update both of them
            self.camera.K[2, 1] *= self.H / ori_H   # update both of them

        self.quad.resize_textures(self.H, self.W)  # maybe not
        for mesh in self.meshes:
            if hasattr(mesh, 'resize_textures'):
                mesh.resize_textures(self.H, self.W)  # maybe not
        gl.glViewport(0, 0, self.W, self.H)
        gl.glScissor(0, 0, self.W, self.H)

    def glfw_error_callback(self, error, description):
        log(red('GLFW Error'), error, description)

    @property
    def window_address(self):
        window_address = ctypes.cast(self.window, ctypes.c_void_p).value
        return window_address

    def bind_callbacks(self):
        glfw.set_window_user_pointer(self.window, self)  # set the user, for retrival
        glfw.set_error_callback(self.glfw_error_callback)
        glfw.set_key_callback(self.window, self.glfw_key_callback)
        glfw.set_char_callback(self.window, self.glfw_char_callback)
        glfw.set_scroll_callback(self.window, self.glfw_scroll_callback)
        glfw.set_cursor_pos_callback(self.window, self.glfw_cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self.glfw_mouse_button_callback)
        glfw.set_framebuffer_size_callback(self.window, self.glfw_framebuffer_size_callback)

    def init_camera(self, camera_cfg: dotdict = dotdict(), view_index: int = None):
        # Everything should have been prepared in the dataset
        # We load the first camera out of it
        dataset = self.dataset
        H, W = self.window_size  # dimesions
        M = max(H, W)

        if self.use_window_focal or not hasattr(dataset, 'Ks'):
            K = torch.as_tensor([
                [M * dataset.focal_ratio, 0, W / 2],  # smaller focal, large fov for a bigger picture
                [0, M * dataset.focal_ratio, H / 2],
                [0, 0, 1],
            ], dtype=torch.float)
        else:
            if view_index is None:
                K = dataset.Ks[0, 0].clone()
                ratio = M / max(dataset.Hs[0, 0], dataset.Ws[0, 0])
            else:
                K = dataset.Ks[view_index, 0].clone()
                ratio = M / max(dataset.Hs[view_index, 0], dataset.Ws[view_index, 0])
            K[:2] *= ratio

        if view_index is None:
            R, T = dataset.Rv.clone(), dataset.Tv.clone()  # intrinsics and extrinsics
        else:
            R, T = dataset.Rs[view_index, 0], dataset.Ts[view_index, 0]

        n, f, t, v = dataset.near, dataset.far, 0, 0  # use 0 for default t
        bounds = dataset.bounds.clone()  # avoids modification
        self.camera = Camera(H, W, K, R, T, n, f, t, v, bounds, **camera_cfg)
        self.camera.front = self.camera.front  # perform alignment correction
        self.snap_camera_index = view_index if view_index is not None else 0

    def init_quad(self):
        from easyvolcap.utils.gl_utils import Quad
        from cuda import cudart
        self.quad = Quad(H=self.H, W=self.W, use_quad_cuda=self.use_quad_cuda, compose=self.compose, compose_power=self.compose_power)  # will blit this texture to screen if rendered

    def init_opengl(self):
        from easyvolcap.utils.gl_utils import common_opengl_options
        import OpenGL.GL as gl
        gl.glViewport(0, 0, self.W, self.H)
        common_opengl_options()

    def init_imgui(self):
        imgui.create_context()
        io = imgui.get_io()

        # io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard  # Enable Keyboard Controls # NOTE: This will make imgui always want to capture keyboard
        # io.config_flags |= imgui.ConfigFlags_.nav_enable_gamepad # Enable Gamepad Controls
        io.config_flags |= imgui.ConfigFlags_.docking_enable  # Enable docking
        # io.config_flags |= imgui.ConfigFlags_.viewports_enable # Enable Multi-Viewport / Platform Windows
        # io.config_viewports_no_auto_merge = True
        # io.config_viewports_no_task_bar_icon = True

        # Setup Dear ImGui style
        imgui.style_colors_dark()
        # imgui.style_colors_classic()

        # When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
        style = imgui.get_style()
        style.tab_rounding = 4.0
        style.grab_rounding = 4.0
        style.child_rounding = 4.0
        style.frame_rounding = 4.0
        style.popup_rounding = 8.0
        style.window_rounding = 8.0
        style.scrollbar_rounding = 4.0
        window_bg_color = style.color_(imgui.Col_.window_bg)
        window_bg_color.w = 1.0
        style.set_color_(imgui.Col_.window_bg, window_bg_color)

        # You need to transfer the window address to imgui.backends.glfw_init_for_opengl
        # proceed as shown below to get it.
        imgui.backends.glfw_init_for_open_gl(self.window_address, True)
        imgui.backends.opengl3_init(self.glsl_version)

        io = imgui.get_io()
        self.default_font = io.fonts.add_font_from_file_ttf(self.font_default, self.font_size)
        self.italic_font = io.fonts.add_font_from_file_ttf(self.font_italic, self.font_size)
        self.bold_font = io.fonts.add_font_from_file_ttf(self.font_bold, self.font_size)

        # # Markdown initialization
        # options = imgui_md.MarkdownOptions()
        # # options.font_options.font_base_path = 'assets/fonts'
        # options.font_options.regular_size = self.font_size
        # imgui_md.initialize_markdown(options=options)
        # imgui_md.get_font_loader_function()() # requires imgui_hello

    def init_glfw(self):
        if not glfw.init():
            log(red('Could not initialize OpenGL context'))
            exit(1)

        # Decide GL+GLSL versions
        # GL 3.3 + GLSL 330
        self.glsl_version = '#version 330'
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)  # // 3.2+ only
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, 1)  # 1 is gl.GL_TRUE

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(self.W, self.H, self.window_title, None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(self.use_vsync)  # disable vsync

        icon = load_image(self.icon_file)
        pixels = (icon * 255).astype(np.uint8)
        height, width = icon.shape[:2]
        glfw.set_window_icon(window, 1, [width, height, pixels])  # set icon for the window

        if not window:
            glfw.terminate()
            log(red('Could not initialize window'))
            raise RuntimeError('Failed to initialize window in glfw')

        self.window = window
        cfg.window = window  # MARK: GLOBAL VARIABLE
