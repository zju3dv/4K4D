from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from imgui_bundle import imgui
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

import glm
import torch
import numpy as np

from os.path import join
from scipy import interpolate
from copy import copy, deepcopy

from glm import vec2, vec3, vec4, mat3, mat4, mat4x3, mat2x3  # This is actually highly optimized

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.math_utils import normalize, affine_inverse
from easyvolcap.utils.data_utils import to_numpy, to_tensor, to_cuda, to_list
from easyvolcap.utils.cam_utils import gen_cubic_spline_interp_func, gen_linear_interp_func


def debug_project(proj: mat4, a: vec3, aa: "imgui.ImVec2"):
    # proj: 4, 4
    # a: 3,
    a: vec4 = proj @ vec4(a, 1.0)  # 4, 4 @ 4 = 4
    if a.w <= 0.02:  # depth should be positive to save upfront
        return False
    else:
        aa.x, aa.y = a.x / a.w, a.y / a.w
        return True


def add_debug_line(proj: mat4, a: vec3, b: vec3, col: np.uint32 = 0xffffffff, thickness: float = 2.0):
    # proj: world to screen transformation matrix: 4, 4
    # a: 3,
    # b: 3,
    from imgui_bundle import imgui
    from easyvolcap.utils.imgui_utils import col2imu32

    draw_list: imgui.ImDrawList = imgui.get_background_draw_list()
    aa, bb = imgui.ImVec2(), imgui.ImVec2()
    if debug_project(proj, a, aa) and debug_project(proj, b, bb):
        draw_list.add_line(aa, bb, col2imu32(col), thickness)


def add_debug_text(proj: mat4, a: vec3, text: str, col: np.uint32 = 0xffffffff):
    # proj: world to screen transformation matrix: 4, 4
    # a: 3,
    # text: str
    from imgui_bundle import imgui
    from easyvolcap.utils.imgui_utils import col2imu32

    draw_list: imgui.ImDrawList = imgui.get_background_draw_list()
    aa = imgui.ImVec2()
    if debug_project(proj, a, aa):
        draw_list.add_text(aa, col2imu32(col), text)


def add_debug_text_2d(aa: "imgui.ImVec2", text: str, col: np.uint32 = 0xff4040ff):
    from imgui_bundle import imgui
    from easyvolcap.utils.imgui_utils import col2imu32

    draw_list: imgui.ImDrawList = imgui.get_background_draw_list()
    draw_list.add_text(aa, col2imu32(col), text)


def visualize_axes(proj: mat4, a: vec3, b: vec3, thickness=3.0, name: str = None):  # bounds in world coordinates
    add_debug_text(proj, vec3(b.x + 0.025, a.y, a.z + 0.045), 'x', 0xccccccff)
    add_debug_text(proj, vec3(a.x, b.y + 0.025, a.z + 0.045), 'y', 0xccccccff)
    add_debug_text(proj, vec3(a.x, a.y, b.z + 0.025 + 0.045), 'z', 0xccccccff)
    add_debug_line(proj, vec3(a.x, a.y, a.z), vec3(b.x, a.y, a.z), 0xff4040ff, thickness=thickness)
    add_debug_line(proj, vec3(a.x, a.y, a.z), vec3(a.x, b.y, a.z), 0x40ff40ff, thickness=thickness)
    add_debug_line(proj, vec3(a.x, a.y, a.z), vec3(a.x, a.y, b.z), 0x4040ffff, thickness=thickness)

    if name is not None: add_debug_text(proj, a + vec3(0.045), str(name), 0xccccccff)  # maybe mark the cameras


def visualize_cube(proj: mat4, a: vec3, b: vec3, thickness=3.0, name: str = None):  # bounds in world coordinates
    add_debug_line(proj, vec3(a.x, a.y, a.z), vec3(b.x, a.y, a.z), 0xff4040ff, thickness=thickness)  # X
    add_debug_line(proj, vec3(a.x, b.y, a.z), vec3(b.x, b.y, a.z), 0xffffffff, thickness=thickness)
    add_debug_line(proj, vec3(a.x, a.y, b.z), vec3(b.x, a.y, b.z), 0xffffffff, thickness=thickness)
    add_debug_line(proj, vec3(a.x, b.y, b.z), vec3(b.x, b.y, b.z), 0xffffffff, thickness=thickness)
    add_debug_line(proj, vec3(a.x, a.y, a.z), vec3(a.x, b.y, a.z), 0x40ff40ff, thickness=thickness)  # Y
    add_debug_line(proj, vec3(b.x, a.y, a.z), vec3(b.x, b.y, a.z), 0xffffffff, thickness=thickness)
    add_debug_line(proj, vec3(a.x, a.y, b.z), vec3(a.x, b.y, b.z), 0xffffffff, thickness=thickness)
    add_debug_line(proj, vec3(b.x, a.y, b.z), vec3(b.x, b.y, b.z), 0xffffffff, thickness=thickness)
    add_debug_line(proj, vec3(a.x, a.y, a.z), vec3(a.x, a.y, b.z), 0x4040ffff, thickness=thickness)  # Z
    add_debug_line(proj, vec3(b.x, a.y, a.z), vec3(b.x, a.y, b.z), 0xffffffff, thickness=thickness)
    add_debug_line(proj, vec3(a.x, b.y, a.z), vec3(a.x, b.y, b.z), 0xffffffff, thickness=thickness)
    add_debug_line(proj, vec3(b.x, b.y, a.z), vec3(b.x, b.y, b.z), 0xffffffff, thickness=thickness)

    if name is not None: add_debug_text(proj, a + vec3(0.045), str(name), 0xffcccccc)  # maybe mark the cameras


def visualize_cameras(proj: mat4, ixt: mat3, c2w: mat4x3, axis_size: float = 0.10, col: np.uint32 = 0x80ffffff, thickness: float = 2.0, name: str = None):
    p = c2w[3]  # third row (corresponding to 3rd column)
    focal = (ixt[0, 0] + ixt[1, 1]) / 2
    axis_size = focal * axis_size / 1000

    aspect = ixt[0, 0] / ixt[1, 1]
    xs = axis_size * aspect
    ys = axis_size
    zs = axis_size * aspect * 2

    a = p + xs * c2w[0] + ys * c2w[1] + zs * c2w[2]
    b = p - xs * c2w[0] + ys * c2w[1] + zs * c2w[2]
    c = p - xs * c2w[0] - ys * c2w[1] + zs * c2w[2]
    d = p + xs * c2w[0] - ys * c2w[1] + zs * c2w[2]

    add_debug_line(proj, p, a, col, thickness)
    add_debug_line(proj, p, b, col, thickness)
    add_debug_line(proj, p, c, col, thickness)
    add_debug_line(proj, p, d, col, thickness)
    add_debug_line(proj, a, b, col, thickness)
    add_debug_line(proj, b, c, col, thickness)
    add_debug_line(proj, c, d, col, thickness)
    add_debug_line(proj, d, a, col, thickness)

    add_debug_line(proj, p, p + axis_size * c2w[0], 0xff4040ff, thickness)
    add_debug_line(proj, p, p + axis_size * c2w[1], 0x40ff40ff, thickness)
    add_debug_line(proj, p, p + axis_size * c2w[2], 0x4040ffff, thickness)

    if name is not None: add_debug_text(proj, p, str(name), 0xccccccff)  # maybe mark the cameras


class CameraPath:
    # This is the Model in the EVC gui designs

    # Basic a list of cameras with interpolations
    # Use the underlying Camera class as backbone
    # Will export to a sequence of cameras extri.yml and intri.yml
    # Will support keyframes based manipulations:
    # 1. Adding current view as key frame
    # 2. Jumping to previous keyframe (resize window as well?) (toggle edit state?) (or just has a replace button)
    #    - Snap to current keyframe?
    #    - Editing would be much better (just a button to replace the selected keyframe)
    # 3. Toggle playing animation of this keyframe (supports some degress of control)
    # 4. Export the animation as a pair of extri.yml and intri.yml
    # 5. imguizmo control of the created camera in the list (translation, rotation etc)
    def __init__(self,
                 playing: bool = False,
                 playing_time: float = 0.5,
                 playing_speed: float = 0.0005,

                 n_render_views: int = 100,
                 render_plots: bool = True,

                 # Visualization related
                 visible: bool = True,
                 name: str = 'camera_path',
                 filename: str = '',
                 plot_thickness: float = 8.0,
                 camera_thickness: float = 6.0,
                 plot_color: int = 0x80ff80ff,
                 camera_color: int = 0x80ffffff,
                 camera_axis_size: float = 0.10,

                 **kwargs,
                 ) -> None:
        self.keyframes: List[Camera] = []  # orders matter
        self.playing_time = playing_time  # range: 0-1

        self.playing = playing  # is this playing? update cam if it is
        self.playing_speed = playing_speed  # faster interpolation time
        self.n_render_views = n_render_views
        self.render_plots = render_plots

        # Private
        self.cursor_index = -1  # the camera to edit
        self.periodic = True

        # Visualization
        self.name = name
        self.visible = visible
        self.plot_thickness = plot_thickness
        self.camera_thickness = camera_thickness
        self.plot_color = plot_color
        self.camera_color = camera_color
        self.camera_axis_size = camera_axis_size
        if filename:
            self.load_keyframes(filename)

    def __len__(self):
        return len(self.keyframes)

    @property
    def loop_interp(self):
        return self.periodic

    @loop_interp.setter
    def loop_interp(self, v: bool):
        changed = self.periodic != v
        self.periodic = v
        if changed: self.update()  # only perform heavy operation after change

    @property
    def selected(self):
        return self.cursor_index

    @selected.setter
    def selected(self, v: int):
        if v >= len(self): return
        if not len(self): self.cursor_index = -1; return
        self.cursor_index = range(len(self))[v]
        denom = (len(self) - 1)
        if denom: self.playing_time = self.cursor_index / denom  # 1 means last frame
        else: self.playing_time = 0.5

    def replace(self, camera: Camera):
        self.keyframes[self.selected] = deepcopy(camera)
        self.update()

    def insert(self, camera: Camera):
        self.keyframes = self.keyframes[:self.selected + 1] + [deepcopy(camera)] + self.keyframes[self.selected + 1:]
        self.selected = self.selected + 1
        self.update()

    def delete(self, index: int):
        del self.keyframes[index]
        self.selected = self.selected - 1  # go back one
        self.update()

    def clear(self):
        self.keyframes.clear()
        self.selected = -1

    def update(self):
        # MARK: HEAVY
        K = len(self.keyframes)
        if K <= 3: return

        # Prepare for linear and extrinsic parameters
        ks = np.asarray([c.K.to_list() for c in self.keyframes]).transpose(0, 2, 1).reshape(K, -1)  # 9
        hs = np.asarray([c.H for c in self.keyframes]).reshape(K, -1)
        ws = np.asarray([c.W for c in self.keyframes]).reshape(K, -1)
        ns = np.asarray([c.n for c in self.keyframes]).reshape(K, -1)
        fs = np.asarray([c.f for c in self.keyframes]).reshape(K, -1)
        ts = np.asarray([c.t for c in self.keyframes]).reshape(K, -1)
        vs = np.asarray([c.v for c in self.keyframes]).reshape(K, -1)
        bs = np.asarray([c.bounds.to_list() for c in self.keyframes]).reshape(K, -1)  # 6
        lins = np.concatenate([ks, hs, ws, ns, fs, ts, vs, bs], axis=-1)  # K, D
        c2ws = np.asarray([c.c2w.to_list() for c in self.keyframes]).transpose(0, 2, 1)  # K, 3, 4

        # Recompute interpolation parameters
        self.lin_func = gen_linear_interp_func(lins, smoothing_term=0.0 if self.periodic else 10.0)  # smoothness: 0 -> period, >0 -> non-period, -1 orbit (not here)
        self.c2w_func = gen_cubic_spline_interp_func(c2ws, smoothing_term=0.0 if self.periodic else 10.0)

    def interp(self, us: float, **kwargs):
        K = len(self.keyframes)
        if K <= 3: return

        # MARK: HEAVY?
        # Actual interpolation
        lin = self.lin_func(us)
        c2w = self.c2w_func(us)

        # Extract linear parameters
        K = torch.as_tensor(lin[:9]).view(3, 3)  # need a transpose
        H = int(lin[9])
        W = int(lin[10])
        n = torch.as_tensor(lin[11], dtype=torch.float)
        f = torch.as_tensor(lin[12], dtype=torch.float)
        t = torch.as_tensor(lin[13], dtype=torch.float)
        v = torch.as_tensor(lin[14], dtype=torch.float)
        bounds = torch.as_tensor(lin[15:]).view(2, 3)  # no need for transpose

        # Extract splined parameters
        w2c = affine_inverse(torch.as_tensor(c2w))  # already float32
        R = w2c[:3, :3]
        T = w2c[:3, 3:]

        return H, W, K, R, T, n, f, t, v, bounds

    def export_keyframes(self, path: str):
        # Store keyframes to path
        cameras = {f'{i:06d}': k.to_easymocap() for i, k in enumerate(self.keyframes)}
        write_camera(cameras, path)  # without extri.yml, only dirname
        log(yellow(f'Keyframes saved to: {blue(path)}'))

    def load_keyframes(self, path: str):
        # Store keyframes to path
        cameras = read_camera(join(path, 'intri.yml'), join(path, 'extri.yml'))
        cameras = dotdict({k: cameras[k] for k in sorted(cameras.keys())})  # assuming dict is ordered (python 3.7+)
        self.keyframes = [Camera().from_easymocap(cam) for cam in cameras.values()]
        self.name = path
        self.update()

    def export_interps(self, path: str):
        # Store interpolations (animation) to path
        us = np.linspace(0, 1, self.n_render_views, dtype=np.float32)

        cameras = dotdict()
        for i, u in enumerate(tqdm(us, desc='Exporting interpolated cameras')):
            cameras[f'{i:06d}'] = self.interp(u).to_easymocap()
        write_camera(cameras, path)  # without extri.yml, only dirname
        log(yellow(f'Interpolated cameras saved to: {blue(path)}'))

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        # from easyvolcap.utils.gl_utils import Mesh
        # Mesh.render_imgui(self, viewer, batch)
        from imgui_bundle import imgui
        from easyvolcap.utils.imgui_utils import push_button_color, pop_button_color, col2rgba, col2vec4, vec42col, list2col, col2imu32

        i = batch.i
        will_delete = batch.will_delete
        slider_width = batch.slider_width

        imgui.push_item_width(slider_width * 0.5)
        self.name = imgui.input_text(f'Mesh name##{i}', self.name)[1]
        self.n_render_views = imgui.slider_int(f'Plot samples##{i}', self.n_render_views, 0, 3000)[1]
        self.plot_thickness = imgui.slider_float(f'Plot thickness##{i}', self.plot_thickness, 0.01, 10.0)[1]
        self.camera_thickness = imgui.slider_float(f'Camera thickness##{i}', self.camera_thickness, 0.01, 10.0)[1]
        self.camera_axis_size = imgui.slider_float(f'Camera axis size##{i}', self.camera_axis_size, 0.01, 1.0)[1]

        self.plot_color = list2col(imgui.color_edit4(f'Plot color##{i}', col2vec4(self.plot_color), flags=imgui.ColorEditFlags_.no_inputs.value)[1])
        self.camera_color = list2col(imgui.color_edit4(f'Camera color##{i}', col2vec4(self.camera_color), flags=imgui.ColorEditFlags_.no_inputs.value)[1])

        push_button_color(0x55cc33ff if not self.render_plots else 0x8855aaff)
        if imgui.button(f'No Plot##{i}' if not self.render_plots else f' Plot ##{i}'):
            self.render_plots = not self.render_plots
        pop_button_color()

        imgui.same_line()
        push_button_color(0x55cc33ff if not self.visible else 0x8855aaff)
        if imgui.button(f'Show##{i}' if not self.visible else f'Hide##{i}'):
            self.visible = not self.visible
        pop_button_color()

        # Render the delete button
        imgui.same_line()
        push_button_color(0xff5533ff)
        if imgui.button(f'Delete##{i}'):
            will_delete.append(i)
        pop_button_color()

        # The actual rendering
        self.draw(viewer.camera)

    def draw(self, camera: Camera):

        # The actual rendering starts here, the camera paths are considered GUI elements for eaiser management
        # This rendering pattern is extremly slow and hard on the CPU, but whatever for now, just visualization
        if not self.visible: return
        if not len(self): return
        proj = camera.w2p  # 3, 4

        # Render cameras
        for i, cam in enumerate(self.keyframes):
            ixt = cam.ixt
            c2w = cam.c2w
            c2w = mat4x3(c2w)  # vis cam only supports this

            # Add to imgui rendering list
            visualize_cameras(proj, ixt, c2w, col=self.camera_color, thickness=self.camera_thickness, axis_size=self.camera_axis_size)

        if self.render_plots and len(self) >= 4:
            us = np.linspace(0, 1, self.n_render_views, dtype=np.float32)
            c2ws = self.c2w_func(us)
            cs = c2ws[..., :3, 3]  # N, 3
            for i, c in enumerate(cs):
                if i == 0:
                    p = c  # previous
                    continue
                add_debug_line(proj, vec3(*p), vec3(*c), col=self.plot_color, thickness=self.plot_thickness)
                p = c

    def render(self, camera: Camera):
        pass


class Camera:
    # Helper class to manage camera parameters
    def __init__(self,
                 H: int = 512,
                 W: int = 512,
                 K: torch.Tensor = torch.tensor([[512.0, 0.0, 256], [0.0, 512.0, 256.0], [0.0, 0.0, 1.0]]),  # intrinsics
                 R: torch.Tensor = torch.tensor([[-0.9977766275405884, 0.06664637476205826, 0.0], [0.004728599451482296, 0.07079283893108368, -0.9974799156188965], [-0.0664784237742424, -0.9952622056007385, -0.07095059007406235]]),  # extrinsics
                 T: torch.Tensor = torch.tensor([[-2.059340476989746e-5], [2.5779008865356445e-6], [-3.000047445297241]]),  # extrinsics
                 n: float = 0.002,  # bounds limit
                 f: float = 100,  # bounds limit
                 t: float = 0.0,  # temporal dimension (implemented as a float instead of int)
                 v: float = 0.0,  # view dimension (implemented as a float instead of int)
                 bounds: torch.Tensor = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),  # bounding box

                 # camera update hyperparameters
                 origin: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 world_up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0]),
                 movement_speed: float = 1.0,  # gui movement speed
                 movement_force: float = 1.0,  # include some physiscs
                 drag_coeff_mult: float = 1.0,  # include some physiscs
                 constant_drag: float = 1.0,
                 mass: float = 0.1,
                 moment_of_inertia: float = 0.1,
                 movement_torque: float = 1.0,
                 angular_friction: float = 2.0,
                 constant_torque: float = 1.0,

                 min_interval: float = 0.0334,  # simulate at at least 30 fps
                 pause_physics: bool = False,

                 batch: dotdict = None,  # will ignore all other inputs
                 string: str = None,  # will ignore all other inputs
                 **kwargs,
                 ) -> None:

        # Batch (network input parameters)
        if string is None:
            if batch is None:
                batch = dotdict()
                batch.H, batch.W, batch.K, batch.R, batch.T, batch.n, batch.f, batch.t, batch.v, batch.bounds = H, W, K, R, T, n, f, t, v, bounds
            self.from_batch(batch)

            # Other configurables
            self.origin = vec3(*origin)
            # self.origin = self.center  # rotate about center
            self.world_up = vec3(*world_up)
            self.movement_speed = movement_speed
            # self.front = self.front  # will trigger an update
        else:
            self.from_string(string)

        # Internal states to facilitate camera position change
        self.is_dragging = False  # rotation
        self.about_origin = False  # about origin rotation
        self.is_panning = False  # translation
        self.lock_fx_fy = True
        self.drag_start = vec2(0.0)

        # Internal states to facilitate moving with mass
        self.mass = mass
        self.force = vec3(0.0)
        self.speed = vec3(0.0)  # no movement
        self.acc = vec3(0.0)
        self.drag_coeff_mult = drag_coeff_mult
        self.movement_force = movement_force
        self.constant_drag = constant_drag
        self.pause_physics = pause_physics
        self.min_interval = min_interval

        self.torque = vec3(0.0)
        self.moment_of_inertia = moment_of_inertia
        self.angular_speed = vec3(0.0)  # relative angular speed on three euler angles
        self.angular_acc = vec3(0.0)
        self.angular_friction = angular_friction
        self.constant_torque = constant_torque
        self.movement_torque = movement_torque

    def step(self, interval: float):
        if self.pause_physics: return

        # Limit interval to make the simulation more stable
        interval = min(interval, self.min_interval)

        # Compute the drag force
        speed2 = glm.dot(self.speed, self.speed)
        if speed2 > 1.0:
            # Drag at opposite direction of movement
            drag = -speed2 * (self.speed / speed2) * self.drag_coeff_mult
        elif speed2 > 0:
            # Constant drag if speed is blow a threshold to make it stop faster
            drag = -self.constant_drag * self.speed
        else:
            drag = vec3(0.0)

        # Compute acceleration and final speed
        self.acc = (self.force + drag) / self.mass
        self.speed += self.acc * interval

        # Compute displacement in this interval
        speed2 = glm.dot(self.speed, self.speed)
        if speed2 > 0:
            direction = mat3(self.right, -glm.normalize(glm.cross(self.right, self.world_up)), self.world_up)
            movement = direction @ (self.speed - self.acc * interval / 2) * interval
            self.center += movement

        # Compute rotation change

        # Compute the drag torque
        speed2 = glm.dot(self.angular_speed, self.angular_speed)
        if speed2 > 0.1:
            # Drag at opposite direction of movement
            drag = -speed2 * (self.angular_speed / speed2) * self.angular_friction
        elif speed2 > 0.0:
            # Constant drag if speed is blow a threshold to make it stop faster
            drag = -self.constant_torque * self.angular_speed
        else:
            drag = vec3(0.0)

        # Compute angular acceleration and final angular speed
        self.angular_acc = (self.torque + drag) / self.moment_of_inertia
        self.angular_speed += self.angular_acc * interval

        # Angular movement direction
        delta = self.angular_speed * interval  # about x, y and z axis (euler angle)

        # Limit look up
        dot = glm.dot(self.world_up, self.front)
        self.drag_ymin = -np.arccos(-dot) + 0.01  # drag up, look down
        self.drag_ymax = np.pi + self.drag_ymin - 0.02  # remove the 0.01 of drag_ymin

        # Rotate about euler angle
        m = mat4(1.0)
        m = glm.rotate(m, np.clip(delta.x, self.drag_ymin, self.drag_ymax), self.right)
        m = glm.rotate(m, delta.y, -self.world_up)
        m = glm.rotate(m, delta.z, self.front)
        center = self.center
        self.front = m @ self.front  # might overshoot and will update center
        self.center = center

    @property
    def w2p(self):
        ixt = mat4(self.ixt)
        ixt[3, 3] = 0
        ixt[2, 3] = 1
        return ixt @ self.ext  # w2c -> c2p = w2p

    @property
    def V(self): return self.c2w

    @property
    def ixt(self): return self.K

    @property
    def gl_ext(self):
        gl_c2w = self.c2w
        gl_c2w[0] *= 1  # do notflip x
        gl_c2w[1] *= -1  # flip y
        gl_c2w[2] *= -1  # flip z
        gl_ext = glm.affineInverse(gl_c2w)
        return gl_ext  # use original opencv ext since we've taken care of the intrinsics in gl_ixt

    @property
    def gl_ixt(self):
        # Construct opengl camera matrix with projection & clipping
        # https://fruty.io/2019/08/29/augmented-reality-with-opencv-and-opengl-the-tricky-projection-matrix/
        # https://gist.github.com/davegreenwood/3a32d779f81f08dce32f3bb423672191
        # fmt: off
        gl_ixt = mat4(
                      2 * self.fx / self.W,                          0,                                       0,  0,
                       2 * self.s / self.W,       2 * self.fy / self.H,                                       0,  0,
                1 - 2 * (self.cx / self.W), 2 * (self.cy / self.H) - 1,   (self.f + self.n) / (self.n - self.f), -1,
                                         0,                          0, 2 * self.f * self.n / (self.n - self.f),  0,
        )
        # fmt: on

        return gl_ixt

    @property
    def ext(self): return self.w2c

    @property
    def w2c(self):
        w2c = mat4(self.R)
        w2c[3] = vec4(*self.T, 1.0)
        return w2c

    @property
    def c2w(self):
        return glm.affineInverse(self.w2c)

    @property
    def right(self) -> vec3: return vec3(self.R[0, 0], self.R[1, 0], self.R[2, 0])  # c2w R, 0 -> 3,

    @property
    def down(self) -> vec3: return vec3(self.R[0, 1], self.R[1, 1], self.R[2, 1])  # c2w R, 1 -> 3,

    @property
    def front(self) -> vec3: return vec3(self.R[0, 2], self.R[1, 2], self.R[2, 2])  # c2w R, 2 -> 3,

    @front.setter
    def front(self, v: vec3):
        front = v  # the last row of R
        self.R[0, 2], self.R[1, 2], self.R[2, 2] = front.x, front.y, front.z
        right = glm.normalize(glm.cross(self.front, self.world_up))  # right
        self.R[0, 0], self.R[1, 0], self.R[2, 0] = right.x, right.y, right.z
        down = glm.cross(self.front, self.right)  # down
        self.R[0, 1], self.R[1, 1], self.R[2, 1] = down.x, down.y, down.z

    @property
    def center(self):
        return -glm.transpose(self.R) @ self.T  # 3,

    @center.setter
    def center(self, v: vec3):
        self.T = -self.R @ v  # 3, 1

    @property
    def s(self): return self.K[1, 0]

    @s.setter
    def s(self, s): self.K[1, 0] = s

    @property
    def fx(self): return self.K[0, 0]

    @fx.setter
    def fx(self, v: float):
        v = min(v, 1e5)
        v = max(v, 1e-3)
        if self.lock_fx_fy:
            self.K[1, 1] = v / self.K[0, 0] * self.K[1, 1]
        self.K[0, 0] = v

    @property
    def fy(self): return self.K[1, 1]

    @fy.setter
    def fy(self, v: float):
        if self.lock_fx_fy:
            self.K[0, 0] = v / self.K[1, 1] * self.K[0, 0]
        self.K[1, 1] = v

    @property
    def cx(self): return self.K[2, 0]

    @cx.setter
    def cx(self, v: float):
        self.K[2, 0] = v

    @property
    def cy(self): return self.K[2, 1]

    @cy.setter
    def cy(self, v: float):
        self.K[2, 1] = v

    def begin_dragging(self,
                       x: float, y: float,
                       is_panning: bool,
                       about_origin: bool,
                       ):
        self.is_dragging = True
        self.is_panning = is_panning
        self.about_origin = about_origin
        self.drag_start = vec2([x, y])

    def end_dragging(self):
        self.is_dragging = False

    def update_dragging(self, x: float, y: float):
        if not self.is_dragging:
            return

        current = vec2(x, y)
        delta = current - self.drag_start
        delta /= max(self.H, self.W)
        delta *= -1

        self.drag_start = vec2([x, y])
        self.drag_start_front = self.front  # a recording
        self.drag_start_down = self.down
        self.drag_start_right = self.right
        self.drag_start_center = self.center
        self.drag_start_origin = self.origin
        self.drag_start_world_up = self.world_up

        # Need to find the max or min delta y to align with world_up
        dot = glm.dot(self.world_up, self.front)
        self.drag_ymin = -np.arccos(-dot) + 0.01  # drag up, look down
        self.drag_ymax = np.pi + self.drag_ymin - 0.02  # remove the 0.01 of drag_ymin

        if self.is_panning:
            delta *= self.movement_speed
            center_delta = delta[0] * self.drag_start_right + delta[1] * self.drag_start_down
            self.center = self.drag_start_center + center_delta
            if self.about_origin:
                self.origin = self.drag_start_origin + center_delta
        else:
            m = mat4(1.0)
            m = glm.rotate(m, delta.x % 2 * np.pi, self.world_up)
            m = glm.rotate(m, np.clip(delta.y, self.drag_ymin, self.drag_ymax), self.drag_start_right)
            self.front = m @ self.drag_start_front  # might overshoot

            if self.about_origin:
                self.center = -m @ (self.origin - self.drag_start_center) + self.origin

    def move(self, x_offset: float, y_offset: float):
        speed_factor = 1e-1
        movement = y_offset * speed_factor
        movement = movement * self.front * self.movement_speed
        self.center += movement

        if self.is_dragging:
            self.drag_start_center += movement

    def to_batch(self):
        meta = dotdict()
        meta.H = torch.as_tensor(self.H)
        meta.W = torch.as_tensor(self.W)
        meta.K = torch.as_tensor(self.K.to_list(), dtype=torch.float).mT
        meta.R = torch.as_tensor(self.R.to_list(), dtype=torch.float).mT
        meta.T = torch.as_tensor(self.T.to_list(), dtype=torch.float)[..., None]
        meta.n = torch.as_tensor(self.n, dtype=torch.float)
        meta.f = torch.as_tensor(self.f, dtype=torch.float)
        meta.t = torch.as_tensor(self.t, dtype=torch.float)
        meta.v = torch.as_tensor(self.v, dtype=torch.float)
        meta.bounds = torch.as_tensor(self.bounds.to_list(), dtype=torch.float)  # no transpose for bounds

        # GUI related elements
        meta.mass = torch.as_tensor(self.mass, dtype=torch.float)
        meta.moment_of_inertia = torch.as_tensor(self.moment_of_inertia, dtype=torch.float)
        meta.movement_force = torch.as_tensor(self.movement_force, dtype=torch.float)
        meta.movement_torque = torch.as_tensor(self.movement_torque, dtype=torch.float)
        meta.movement_speed = torch.as_tensor(self.movement_speed, dtype=torch.float)
        meta.origin = torch.as_tensor(self.origin.to_list(), dtype=torch.float)
        meta.world_up = torch.as_tensor(self.world_up.to_list(), dtype=torch.float)

        batch = dotdict()
        batch.update(meta)
        batch.meta.update(meta)
        return batch

    def to_easymocap(self):
        batch = self.to_batch()
        camera = to_numpy(batch)
        return camera

    def from_easymocap(self, camera: dict):
        batch = to_tensor(camera)
        self.from_batch(batch)
        return self

    def to_string(self) -> str:
        batch = to_list(self.to_batch().meta)
        return json.dumps(batch)

    def from_string(self, string: str):
        batch = to_tensor(dotdict(json.loads(string)), ignore_list=True)
        self.from_batch(batch)

    def from_batch(self, batch: dotdict):
        H, W, K, R, T, n, f, t, v, bounds = batch.H, batch.W, batch.K, batch.R, batch.T, batch.n, batch.f, batch.t, batch.v, batch.bounds

        # Batch (network input parameters)
        self.H = int(H)
        self.W = int(W)
        self.K = mat3(*K.mT.ravel())
        self.R = mat3(*R.mT.ravel())
        self.T = vec3(*T.ravel())  # 3,
        self.n = float(n)
        self.f = float(f)
        self.t = float(t)
        self.v = float(v)
        self.bounds = mat2x3(*bounds.ravel())  # 2, 3

        if 'mass' in batch: self.mass = float(batch.mass)
        if 'moment_of_inertia' in batch: self.moment_of_inertia = float(batch.moment_of_inertia)
        if 'movement_force' in batch: self.movement_force = float(batch.movement_force)
        if 'movement_torque' in batch: self.movement_torque = float(batch.movement_torque)
        if 'movement_speed' in batch: self.movement_speed = float(batch.movement_speed)
        if 'origin' in batch: self.origin = vec3(*batch.origin.ravel())  # 3,
        if 'world_up' in batch: self.world_up = vec3(*batch.world_up.ravel())  # 3,
        return self

    def custom_pose(self, R: torch.Tensor, T: torch.Tensor, K: torch.Tensor):
        # self.K = mat3(*K.mT.ravel())
        self.R = mat3(*R.mT.ravel())
        self.T = vec3(*T.ravel())
