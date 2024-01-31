from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.utils.egl_utils import create_opengl_context, eglContextManager  # must be imported before OpenGL.GL
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

import os
import sys
import glm
import torch
import ctypes
import numpy as np

from torch import nn
from enum import Enum, auto
from types import MethodType
from typing import Dict, Union, List
from glm import vec2, vec3, vec4, mat3, mat4, mat4x3, mat2x3  # This is actually highly optimized

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.viewer_utils import Camera
from easyvolcap.utils.bound_utils import get_bounds
from easyvolcap.utils.chunk_utils import multi_gather
from easyvolcap.utils.color_utils import cm_cpu_store
from easyvolcap.utils.ray_utils import create_meshgrid
from easyvolcap.utils.depth_utils import depth_curve_fn
from easyvolcap.utils.gaussian_utils import rgb2sh0, sh02rgb
from easyvolcap.utils.nerf_utils import volume_rendering, raw2alpha
from easyvolcap.utils.data_utils import load_pts, load_mesh, to_cuda, add_batch
from easyvolcap.utils.cuda_utils import CHECK_CUDART_ERROR, FORMAT_CUDART_ERROR
from easyvolcap.utils.net_utils import typed, torch_dtype_to_numpy_dtype, load_pretrained
from easyvolcap.utils.fcds_utils import prepare_feedback_transform, get_opencv_camera_params


# fmt: off
# Environment variable messaging
# Need to export EGL_DEVICE_ID before trying to import egl
# And we need to consider the case when we're performing distributed training
# from easyvolcap.engine import cfg, args  # FIXME: GLOBAL IMPORTS
if 'easyvolcap.engine' in sys.modules and \
    (sys.modules['easyvolcap.engine'].args.type != 'gui' or \
        sys.modules['easyvolcap.engine'].cfg.viewer_cfg.type != 'VolumetricVideoViewer'): # FIXME: GLOBAL VARIABLES
    try:
        from easyvolcap.utils.egl_utils import create_opengl_context, eglContextManager
    except Exception as e:
        log(yellow(f'Could not import EGL related modules. {type(e).__name__}: {e}'))
        os.environ['PYOPENGL_PLATFORM'] = ''

def is_wsl2():
    """Returns True if the current environment is WSL2, False otherwise."""
    return exists("/etc/wsl.conf") and os.environ.get("WSL_DISTRO_NAME")

if is_wsl2():
    os.environ['PYOPENGL_PLATFORM'] = 'glx'

import OpenGL.GL as gl

try:
    from OpenGL.GL import shaders
except Exception as e:
    print(f'WARNING: OpenGL shaders import error encountered, please install the latest PyOpenGL from github using:')
    print(f'pip install git+https://github.com/mcfletch/pyopengl')
    raise e
# fmt: on


def linearize_depth(d, n: float, f: float):
    # 0-1 -> -1,1
    # ndc -> view
    return (2.0 * n * f) / (f + n - (d * 2 - 1) * (f - n))


def common_opengl_options():
    # Use program point size
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    # Performs face culling
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glCullFace(gl.GL_BACK)

    # Performs alpha trans testing
    # gl.glEnable(gl.GL_ALPHA_TEST)
    try: gl.glEnable(gl.GL_ALPHA_TEST)
    except gl.GLError as e: pass

    # Performs z-buffer testing
    gl.glEnable(gl.GL_DEPTH_TEST)
    # gl.glDepthMask(gl.GL_TRUE)
    gl.glDepthFunc(gl.GL_LEQUAL)
    # gl.glDepthRange(-1.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Enable some masking tests
    gl.glEnable(gl.GL_SCISSOR_TEST)

    # Enable this to correctly render points
    # https://community.khronos.org/t/gl-point-sprite-gone-in-3-2/59310
    # gl.glEnable(gl.GL_POINT_SPRITE)  # MARK: ONLY SPRITE IS WORKING FOR NOW
    try: gl.glEnable(gl.GL_POINT_SPRITE)  # MARK: ONLY SPRITE IS WORKING FOR NOW
    except gl.GLError as e: pass
    # gl.glEnable(gl.GL_POINT_SMOOTH) # MARK: ONLY SPRITE IS WORKING FOR NOW

    # # Configure how we store the pixels in memory for our subsequent reading of the FBO to store the rendering into memory.
    # # The second argument specifies that our pixels will be in bytes.
    # gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)


def load_shader_source(file: str = 'splat.frag'):
    # Ideally we can just specify the shader name instead of an variable
    if not exists(file):
        file = f'{dirname(__file__)}/shaders/{file}'
    if not exists(file):
        file = file.replace('shaders/', '')
    if not exists(file):
        raise RuntimeError(f'Shader file: {file} does not exist')
    with open(file, 'r') as f:
        return f.read()


def use_gl_program(program: Union[shaders.ShaderProgram, dict]):
    if isinstance(program, dict):
        # Recompile the program if the user supplied sources
        program = dotdict(program)
        program = shaders.compileProgram(
            shaders.compileShader(program.VERT_SHADER_SRC, gl.GL_VERTEX_SHADER),
            shaders.compileShader(program.FRAG_SHADER_SRC, gl.GL_FRAGMENT_SHADER)
        )
    return gl.glUseProgram(program)


class Mesh:
    class RenderType(Enum):
        POINTS = 1
        LINES = 2
        TRIS = 3
        QUADS = 4  # TODO: Support quad loading
        STRIPS = 5

    # Helper class to render a mesh on opengl
    # This implementation should only be used for debug visualization
    # Since no differentiable mechanism will be added
    # We recommend using nvdiffrast and pytorch3d's point renderer directly if you will to optimize these structures directly

    def __init__(self,
                 verts: torch.Tensor = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),  # need to call update after update
                 faces: torch.Tensor = torch.tensor([[0, 1, 2]]),  # need to call update after update
                 colors: torch.Tensor = None,
                 normals: torch.Tensor = None,
                 scalars: dotdict[str, torch.Tensor] = dotdict(),
                 render_type: RenderType = RenderType.TRIS,

                 # Misc info
                 name: str = 'mesh',
                 filename: str = '',
                 visible: bool = True,

                 # Render options
                 shade_flat: bool = False,  # smooth shading
                 point_radius: float = 0.015,
                 render_normal: bool = False,

                 # Storage options
                 store_device: str = 'cpu',
                 compute_device: str = 'cuda',
                 vert_sizes=[3, 3, 3],  # pos + color + norm

                 # Init options
                 est_normal_thresh: int = 100000,

                 # Ignore unused input
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.name = name
        self.visible = visible
        self.render_type = render_type

        self.shade_flat = shade_flat
        self.point_radius = point_radius
        self.render_normal = render_normal

        self.store_device = store_device
        self.compute_device = compute_device
        self.vert_sizes = vert_sizes

        self.est_normal_thresh = est_normal_thresh

        # Uniform and program
        self.compile_shaders()
        self.uniforms = dotdict()  # uniform values

        # Before initialization
        self.max_verts = 0
        self.max_faces = 0

        # OpenGL data
        if filename: self.load_from_file(filename)
        else: self.load_from_data(verts, faces, colors, normals, scalars)

    def compile_shaders(self):
        try:
            self.mesh_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('mesh.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('mesh.frag'), gl.GL_FRAGMENT_SHADER)
            )
            self.point_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('point.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('point.frag'), gl.GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    @property
    def n_verts_bytes(self):
        return len(self.verts) * self.vert_size * self.verts.element_size()

    @property
    def n_faces_bytes(self):
        return len(self.faces) * self.face_size * self.faces.element_size()

    @property
    def verts_data(self):  # a heavy copy operation
        verts = torch.cat([self.verts, self.colors, self.normals], dim=-1).ravel().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=np.float32, order='C')
        return verts

    @property
    def faces_data(self):  # a heavy copy operation
        faces = self.faces.ravel().numpy()  # N, 3
        faces = np.asarray(faces, dtype=np.uint32, order='C')
        return faces

    @property
    def face_size(self):
        return self.render_type.value

    @property
    def vert_size(self):
        return sum(self.vert_sizes)

    def load_from_file(self, filename: str = 'assets/meshes/bunny.ply'):
        verts, faces, colors, normals, scalars = self.load_data_from_file(filename)
        self.load_from_data(verts, faces, colors, normals, scalars)

    def load_data_from_file(self, filename: str = 'assets/meshes/bunny.ply'):
        self.name = os.path.split(filename)[-1]
        verts, faces, colors, normals, scalars = None, None, None, None, None
        verts, faces = load_mesh(filename, device=self.store_device)
        if not len(faces):
            verts, colors, normals, scalars = load_pts(filename)
            self.render_type = Mesh.RenderType.POINTS
        else:
            self.render_type = Mesh.RenderType(faces.shape[-1])  # use value
        return verts, faces, colors, normals, scalars

    def load_from_data(self, verts: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor = None, normals: torch.Tensor = None, scalars: dotdict[str, torch.Tensor] = dotdict()):
        # Data type conversion
        verts = torch.as_tensor(verts)  # convert to tensor if input is of other types
        if verts.dtype == torch.float32:
            pass  # supports this for now
        elif verts.dtype == torch.float16:
            pass  # supports this for now
        else:
            verts = verts.type(torch.float)  # convert to float32 if input is of higher precision
        gl_dtype = gl.GL_FLOAT if verts.dtype == torch.float else gl.GL_HALF_FLOAT
        self.vert_gl_types = [gl_dtype] * len(self.vert_sizes)

        # Prepare main mesh data: vertices and faces
        self.verts = torch.as_tensor(verts, device=self.store_device)
        self.faces = torch.as_tensor(faces, device=self.store_device, dtype=torch.int32)  # NOTE: No uint32 support

        # Prepare colors and normals
        if colors is not None:
            self.colors = torch.as_tensor(colors, device=self.store_device, dtype=self.verts.dtype)
        else:
            bounds = get_bounds(self.verts[None])[0]
            self.colors = (self.verts - bounds[0]) / (bounds[1] - bounds[0])
        if normals is not None:
            self.normals = torch.as_tensor(normals, device=self.store_device, dtype=self.verts.dtype)
        else:
            self.estimate_vertex_normals()

        # Prepare other scalars
        if scalars is not None:
            for k, v in scalars.items():
                setattr(self, k, torch.as_tensor(v, device=self.store_device, dtype=self.verts.dtype))  # is this ok?

        # Prepare OpenGL related buffer
        self.update_gl_buffers()

    def estimate_vertex_normals(self):
        def est_pcd_norms():
            if self.verts.dtype == torch.half:
                self.normals = self.verts
            else:
                from pytorch3d.structures import Pointclouds, Meshes
                pcd = Pointclouds([self.verts]).to(self.compute_device)
                self.normals = pcd.estimate_normals()[0].cpu().to(self.verts.dtype)  # no batch dim

        def est_tri_norms():
            if self.verts.dtype == torch.half:
                self.normals = self.verts
            else:
                from pytorch3d.structures import Pointclouds, Meshes
                mesh = Meshes([self.verts], [self.faces]).to(self.compute_device)
                self.normals = mesh.verts_normals_packed().cpu().to(self.verts.dtype)  # no batch dim

        if not len(self.verts) > self.est_normal_thresh:
            if self.render_type == Mesh.RenderType.TRIS: est_tri_norms()
            elif self.render_type == Mesh.RenderType.POINTS: est_pcd_norms()
            else:
                # log(yellow(f'Unsupported mesh type: {self.render_type} for normal estimation, skipping'))
                self.normals = self.verts
        else:
            # log(yellow(f'Number of points for mesh too large: {len(self.verts)} > {self.est_normal_thresh}, skipping normal estimation'))
            self.normals = self.verts

    def offscreen_render(self, eglctx: "eglContextManager", camera: Camera):
        eglctx.resize(camera.W, camera.H)
        self.render(camera)

    def render(self, camera: Camera):
        if not self.visible: return

        # For point rendering
        if self.render_type == Mesh.RenderType.POINTS:
            gl.glUseProgram(self.point_program)
            self.use_gl_program(self.point_program)
        else:
            gl.glUseProgram(self.mesh_program)
            self.use_gl_program(self.mesh_program)

        self.upload_gl_uniforms(camera)
        gl.glBindVertexArray(self.vao)

        if self.render_type == Mesh.RenderType.POINTS:
            gl.glDrawArrays(gl.GL_POINTS, 0, len(self.verts))  # number of vertices
        elif self.render_type == Mesh.RenderType.LINES:
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glDrawElements(gl.GL_LINES, len(self.faces) * self.face_size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # number of indices
        elif self.render_type == Mesh.RenderType.TRIS:
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.faces) * self.face_size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # number of indices
        elif self.render_type == Mesh.RenderType.QUADS:
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glDrawElements(gl.GL_QUADS, len(self.faces) * self.face_size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # number of indices
        elif self.render_type == Mesh.RenderType.STRIPS:
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.verts))
        else:
            raise NotImplementedError

        gl.glBindVertexArray(0)

    def use_gl_program(self, program: shaders.ShaderProgram):
        use_gl_program(program)
        self.uniforms.shade_flat = gl.glGetUniformLocation(program, "shade_flat")
        self.uniforms.point_radius = gl.glGetUniformLocation(program, "point_radius")
        self.uniforms.render_normal = gl.glGetUniformLocation(program, "render_normal")
        self.uniforms.H = gl.glGetUniformLocation(program, "H")
        self.uniforms.W = gl.glGetUniformLocation(program, "W")
        self.uniforms.n = gl.glGetUniformLocation(program, "n")
        self.uniforms.f = gl.glGetUniformLocation(program, "f")
        self.uniforms.P = gl.glGetUniformLocation(program, "P")
        self.uniforms.K = gl.glGetUniformLocation(program, "K")
        self.uniforms.V = gl.glGetUniformLocation(program, "V")
        self.uniforms.M = gl.glGetUniformLocation(program, "M")

    def upload_gl_uniforms(self, camera: Camera):
        K = camera.gl_ixt  # hold the reference
        V = camera.gl_ext  # hold the reference
        M = glm.identity(mat4)
        P = K * V * M

        gl.glUniform1i(self.uniforms.shade_flat, self.shade_flat)
        gl.glUniform1f(self.uniforms.point_radius, self.point_radius)
        gl.glUniform1i(self.uniforms.render_normal, self.render_normal)
        gl.glUniform1i(self.uniforms.H, camera.H)  # o2w
        gl.glUniform1i(self.uniforms.W, camera.W)  # o2w
        gl.glUniform1f(self.uniforms.n, camera.n)  # o2w
        gl.glUniform1f(self.uniforms.f, camera.f)  # o2w
        gl.glUniformMatrix4fv(self.uniforms.P, 1, gl.GL_FALSE, glm.value_ptr(P))  # o2clip
        gl.glUniformMatrix4fv(self.uniforms.K, 1, gl.GL_FALSE, glm.value_ptr(K))  # c2clip
        gl.glUniformMatrix4fv(self.uniforms.V, 1, gl.GL_FALSE, glm.value_ptr(V))  # w2c
        gl.glUniformMatrix4fv(self.uniforms.M, 1, gl.GL_FALSE, glm.value_ptr(M))  # o2w

    def update_gl_buffers(self):
        # Might be overwritten
        self.resize_buffers(len(self.verts) if hasattr(self, 'verts') else 0,
                            len(self.faces) if hasattr(self, 'faces') else 0)  # maybe repeated

        if hasattr(self, 'verts'):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.n_verts_bytes, self.verts_data)  # hold the reference
        if hasattr(self, 'faces'):
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferSubData(gl.GL_ELEMENT_ARRAY_BUFFER, 0, self.n_faces_bytes, self.faces_data)

    def resize_buffers(self, v: int = 0, f: int = 0):
        if v > self.max_verts or f > self.max_faces:
            if v > self.max_verts: self.max_verts = v
            if f > self.max_faces: self.max_faces = f
            self.init_gl_buffers(v, f)

    def init_gl_buffers(self, v: int = 0, f: int = 0):
        # This will only init the corresponding buffer object
        n_verts_bytes = v * self.vert_size * self.verts.element_size() if v > 0 else self.n_verts_bytes
        n_faces_bytes = f * self.face_size * self.faces.element_size() if f > 0 else self.n_faces_bytes

        # Housekeeping
        if hasattr(self, 'vao'):
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(2, [self.vbo, self.ebo])

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, n_verts_bytes, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW)  # NOTE: Using pointers here won't work

        # https://stackoverflow.com/questions/67195932/pyopengl-cannot-render-any-vao
        cumsum = 0
        for i, (s, t) in enumerate(zip(self.vert_sizes, self.vert_gl_types)):
            gl.glVertexAttribPointer(i, s, t, gl.GL_FALSE, self.vert_size * self.verts.element_size(), ctypes.c_void_p(cumsum * self.verts.element_size()))  # we use 32 bit float
            gl.glEnableVertexAttribArray(i)
            cumsum += s

        if n_faces_bytes > 0:
            # Some implementation has no faces, we dangerously ignore ebo here, assuming they will never be used
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, n_faces_bytes, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW)
            gl.glBindVertexArray(0)

    def render_imgui(mesh, viewer: 'VolumetricVideoViewer', batch: dotdict):
        from imgui_bundle import imgui
        from easyvolcap.utils.imgui_utils import push_button_color, pop_button_color

        i = batch.i
        will_delete = batch.will_delete
        slider_width = batch.slider_width

        imgui.push_item_width(slider_width * 0.5)
        mesh.name = imgui.input_text(f'Mesh name##{i}', mesh.name)[1]

        if imgui.begin_combo(f'Mesh type##{i}', mesh.render_type.name):
            for t in Mesh.RenderType:
                if imgui.selectable(t.name, mesh.render_type == t)[1]:
                    mesh.render_type = t  # construct enum from name
                if mesh.render_type == t:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_item_width()

        if hasattr(mesh, 'point_radius'):
            mesh.point_radius = imgui.slider_float(f'Point radius##{i}', mesh.point_radius, 0.0005, 3.0)[1]  # 0.1mm

        if hasattr(mesh, 'pts_per_pix'):
            mesh.pts_per_pix = imgui.slider_int('Point per pixel', mesh.pts_per_pix, 0, 60)[1]  # 0.1mm

        if hasattr(mesh, 'shade_flat'):
            push_button_color(0x55cc33ff if not mesh.shade_flat else 0x8855aaff)
            if imgui.button(f'Smooth##{i}' if not mesh.shade_flat else f' Flat ##{i}'):
                mesh.shade_flat = not mesh.shade_flat
            pop_button_color()

        if hasattr(mesh, 'render_normal'):
            imgui.same_line()
            push_button_color(0x55cc33ff if not mesh.render_normal else 0x8855aaff)
            if imgui.button(f'Color ##{i}' if not mesh.render_normal else f'Normal##{i}'):
                mesh.render_normal = not mesh.render_normal
            pop_button_color()

        if hasattr(mesh, 'visible'):
            imgui.same_line()
            push_button_color(0x55cc33ff if not mesh.visible else 0x8855aaff)
            if imgui.button(f'Show##{i}' if not mesh.visible else f'Hide##{i}'):
                mesh.visible = not mesh.visible
            pop_button_color()

        # Render the delete button
        imgui.same_line()
        push_button_color(0xff5533ff)
        if imgui.button(f'Delete##{i}'):
            will_delete.append(i)
        pop_button_color()


class Quad(Mesh):
    # A shared texture for CUDA (pytorch) and OpenGL
    # Could be rendererd to screen using blitting or just drawing a quad
    def __init__(self,
                 H: int = 256, W: int = 256,
                 use_quad_draw: bool = True,
                 use_quad_cuda: bool = True,
                 compose: bool = False,
                 compose_power: float = 1.0,
                 ):  # the texture to blip
        self.use_quad_draw = use_quad_draw
        self.use_quad_cuda = use_quad_cuda
        self.vert_sizes = [3]  # only position
        self.vert_gl_types = [gl.GL_FLOAT]  # only position
        self.render_type = Mesh.RenderType.STRIPS  # remove side effects of settings _type
        self.max_verts, self.max_faces = 0, 0
        self.verts = torch.as_tensor([[-1., -1., 0.5],
                                      [1., -1., 0.5],
                                      [-1., 1., 0.5],
                                      [1., 1., 0.5],])
        self.update_gl_buffers()
        self.compile_shaders()

        self.max_H, self.max_W = H, W
        self.H, self.W = H, W
        self.compose = compose
        self.compose_power = compose_power

        self.init_texture()

    @property
    def n_faces_bytes(self): return 0

    def use_gl_program(self, program: shaders.ShaderProgram):
        super().use_gl_program(program)
        self.uniforms.tex = gl.glGetUniformLocation(program, 'tex')
        gl.glUseProgram(self.quad_program)  # use a different program
        gl.glUniform1i(self.uniforms.tex, 0)

    def compile_shaders(self):
        try:
            self.quad_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('quad.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('quad.frag'), gl.GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    def resize_textures(self, H: int, W: int):  # analogy to update_gl_buffers
        self.H, self.W = H, W
        if self.H > self.max_H or self.W > self.max_W:  # max got updated
            self.max_H, self.max_W = max(int(self.H * 1.05), self.max_H), max(int(self.W * 1.05), self.max_W)
            self.init_texture()

    def init_texture(self):
        if hasattr(self, 'cu_tex'):
            from cuda import cudart
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_tex))

        if hasattr(self, 'fbo'):
            gl.glDeleteFramebuffers(1, [self.fbo])
            gl.glDeleteTextures(1, [self.tex])

        # Init the texture to be blit onto the screen
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self.max_W, self.max_H, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        # Init the framebuffer object if explicit blitting is used (slower than drawing quad)
        self.fbo = gl.glGenFramebuffers(1)
        old_fbo = gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.tex, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, old_fbo)

        if self.use_quad_cuda:
            from cuda import cudart
            if self.compose:
                # Both reading and writing of this resource is required
                flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone
            else:
                flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
            try:
                self.cu_tex = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.tex, gl.GL_TEXTURE_2D, flags))
            except RuntimeError as e:
                log(red('Failed to initialize Quad with CUDA-GL interop, will use slow upload: '), e)
                self.use_quad_cuda = False

    def copy_to_texture(self, image: torch.Tensor, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        if not self.use_quad_cuda:
            self.upload_to_texture(image)
            return

        if not hasattr(self, 'cu_tex'):
            self.init_texture()

        # assert self.use_quad_cuda, "Need to enable cuda-opengl interop to copy from device to device, check creation of this Quad"
        w = w or self.W
        h = h or self.H
        if image.shape[-1] == 3:
            image = torch.cat([image, image.new_ones(image.shape[:-1] + (1,)) * 255], dim=-1)  # add alpha channel

        from cuda import cudart
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.cu_tex, torch.cuda.current_stream().cuda_stream))
        cu_tex_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(self.cu_tex, 0, 0))

        if self.compose:
            """
            Blit current framebuffer to this texture (self.tex)
            Read content of this texture into a cuda buffer
            Perform alpha blending based on the frame's alpha channel
            Copy the blended image back into the texture (self.tex)
            """
            old = gl.glGetInteger(gl.GL_DRAW_FRAMEBUFFER_BINDING)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.fbo)  # read buffer defaults to 0
            gl.glBlitFramebuffer(x, y, w, h,
                                 x, y, w, h,
                                 gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)  # now self.tex contains the content of the already rendered frame
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, old)

            buffer = torch.empty_like(image)
            CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(buffer.data_ptr(),  # dst
                                                                 w * 4 * buffer.element_size(),  # dpitch
                                                                 cu_tex_arr,  # src
                                                                 x * 4 * image.element_size(),  # wOffset
                                                                 y,  # hOffset
                                                                 w * 4 * buffer.element_size(),  # width Width of matrix transfer (columns in bytes)
                                                                 h,  # height
                                                                 kind,  # kind
                                                                 torch.cuda.current_stream().cuda_stream))  # stream

            # cv2.imwrite('image.png', image.flip(0).detach().cpu().numpy()[..., [2,1,0,3]])
            alpha = image[..., -1:] / 255
            image[..., :-1] = buffer[..., :-1] * (1 - alpha ** self.compose_power) + image[..., :-1] * alpha  # storing float into int
            image[..., -1:] = buffer[..., -1:] + image[..., -1:]
            image = image.clip(0, 255)

        CHECK_CUDART_ERROR(cudart.cudaMemcpy2DToArrayAsync(cu_tex_arr,
                                                           x * 4 * image.element_size(),
                                                           y,
                                                           image.data_ptr(),
                                                           w * 4 * image.element_size(),  # differently sized
                                                           w * 4 * image.element_size(),  # rgba, should do a composition first
                                                           h,
                                                           kind,
                                                           torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.cu_tex, torch.cuda.current_stream().cuda_stream))

    def upload_to_texture(self, ptr: np.ndarray, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        w = w or self.W
        h = h or self.H
        if isinstance(ptr, torch.Tensor):
            ptr = ptr.detach().cpu().numpy()  # slow sync and copy operation # MARK: SYNC

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, x, y, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, ptr[y:h, x:w])  # to gpu, might slow down?

    @property
    def verts_data(self):  # a heavy copy operation
        verts = self.verts.ravel().detach().cpu().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=np.float32, order='C')
        return verts

    def render(self, camera: Camera = None):
        self.draw()  # no uploading needed

    def draw(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        """
        Upload the texture instead of the camera
        This respects the OpenGL convension of lower left corners
        """
        if not self.use_quad_draw:
            self.blit(x, y, w, h)
            return

        w = w or self.W
        h = h or self.H
        _, _, W, H = gl.glGetIntegerv(gl.GL_VIEWPORT)
        gl.glViewport(x, y, w, h)
        gl.glScissor(x, y, w, h)  # only render in this small region of the viewport

        gl.glUseProgram(self.quad_program)  # use a different program
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.verts))
        gl.glBindVertexArray(0)

        # Some house keepings
        gl.glViewport(0, 0, W, H)
        gl.glScissor(0, 0, W, H)

    def blit(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        """
        This respects the OpenGL convension of lower left corners
        """
        w = w or self.W
        h = h or self.H
        old = gl.glGetInteger(gl.GL_READ_FRAMEBUFFER_BINDING)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo)  # write buffer defaults to 0
        gl.glBlitFramebuffer(x, y, x + w, y + h,  # the height is flipped
                             x, y, x + w, y + h,  # the height is flipped
                             gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, old)


class UQuad(Mesh):
    """
    Responsible for initializing textures with a single value
    or blitting a texture to a framebuffer (possibly better done with blit instead of quad drawing)
    Effectively clearing the texture for real, see: https://stackoverflow.com/questions/37335281/is-glcleargl-color-buffer-bit-preferred-before-a-whole-frame-buffer-overwritte
    """

    def __init__(self):
        self.n_blit_values = 3
        self.vert_sizes = [3]  # only position
        self.vert_gl_types = [gl.GL_FLOAT]  # only position
        self.max_verts, self.max_faces = 0, 0
        self.verts = torch.as_tensor([[-1., -1., 0.5],
                                      [1., -1., 0.5],
                                      [-1., 1., 0.5],
                                      [1., 1., 0.5],])
        self.compile_shaders()
        self.uniforms = dotdict()  # uniform values
        self.use_gl_programs(self.quad_program)
        self.update_gl_buffers()

    @property
    def n_faces_bytes(self): return 0

    @property
    def verts_data(self):  # a heavy copy operation
        verts = self.verts.ravel().detach().cpu().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=np.float32, order='C')
        return verts

    def use_gl_programs(self, program: shaders.ShaderProgram):
        for i in range(self.n_blit_values):
            self.uniforms[f'value{i}'] = gl.glGetUniformLocation(program, f'value{i}')
        for i in range(self.n_blit_values):
            self.uniforms[f'use_tex{i}'] = gl.glGetUniformLocation(program, f'use_tex{i}')

        gl.glUseProgram(self.program)  # use a different program

        for i in range(self.n_blit_values):
            self.uniforms[f'tex{i}'] = gl.glGetUniformLocation(program, f'tex{i}')
            gl.glUniform1i(self.uniforms[f'tex{i}'], i)

    def upload_gl_uniforms(self, values: List[List[float]], use_texs: List[bool]):
        for i, v in enumerate(values):
            v = vec4(v)  # HACK: Hold the reference for this upload
            gl.glUniform4fv(self.uniforms[f'value{i}'], 1, glm.value_ptr(v))  # as float array
        for i, v in enumerate(use_texs):
            gl.glUniform1i(self.uniforms[f'use_tex{i}'], v)

    def compile_shaders(self):
        try:
            self.quad_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('uquad.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('uquad.frag'), gl.GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    def draw(self, values: List[List[float]] = [], use_texs=[]):
        """
        This function will render 'value' to the currently bound framebuffer, up to six outputs
        """
        old_prog = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        old_vao = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        gl.glUseProgram(self.quad_program)
        self.upload_gl_uniforms(values, use_texs)  # should be a noop

        # Prepare to render to textures
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.verts))  # number of vertices
        gl.glBindVertexArray(old_vao)
        gl.glUseProgram(old_prog)


class DQuad(UQuad):
    def compile_shaders(self):
        try:
            self.quad_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('dquad.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('dquad.frag'), gl.GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    def draw(self, values: List[List[float]] = [], use_texs=[]):
        old_function = gl.glGetIntegerv(gl.GL_DEPTH_FUNC)
        gl.glDepthFunc(gl.GL_ALWAYS)
        super().draw(values, use_texs)
        gl.glDepthFunc(old_function)


def hardware_rendering_framebuffer(H: int, W: int, gl_tex_dtype=gl.GL_RGBA16F):
    # Prepare for write frame buffers
    color_buffer = gl.glGenTextures(1)
    depth_upper = gl.glGenTextures(1)
    depth_lower = gl.glGenTextures(1)
    depth_attach = gl.glGenTextures(1)
    fbo = gl.glGenFramebuffers(1)  # generate 1 framebuffer, storereference in fb

    # Init the texture (call the resizing function), will simply allocate empty memory
    # The internal format describes how the texture shall be stored in the GPU. The format describes how the format of your pixel data in client memory (together with the type parameter).
    gl.glBindTexture(gl.GL_TEXTURE_2D, color_buffer)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl_tex_dtype, W, H, 0, gl.GL_RGBA, gl.GL_FLOAT, ctypes.c_void_p(0))  # 16 * 4
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_upper)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, W, H, 0, gl.GL_RED, gl.GL_FLOAT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_lower)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, W, H, 0, gl.GL_RED, gl.GL_FLOAT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_attach)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, W, H, 0, gl.GL_DEPTH_COMPONENT, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    # Bind texture to fbo
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_buffer, 0)  # location 0
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_TEXTURE_2D, depth_upper, 0)  # location 1
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT2, gl.GL_TEXTURE_2D, depth_lower, 0)  # location 1
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, depth_attach, 0)
    gl.glDrawBuffers(3, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2])

    # Check framebuffer status
    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
        log(red('Framebuffer not complete, exiting...'))
        raise RuntimeError('Incomplete framebuffer')

    # Restore the original state
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return color_buffer, depth_upper, depth_lower, depth_attach, fbo


def hareward_peeling_framebuffer(H: int, W: int):
    # Prepare for write frame buffers
    index_buffer = gl.glGenTextures(1)
    depth_lower = gl.glGenTextures(1)
    depth_attach = gl.glGenTextures(1)
    fbo = gl.glGenFramebuffers(1)  # generate 1 framebuffer, storereference in fb

    # Init the texture (call the resizing function), will simply allocate empty memory
    # The internal format describes how the texture shall be stored in the GPU. The format describes how the format of your pixel data in client memory (together with the type parameter).
    gl.glBindTexture(gl.GL_TEXTURE_2D, index_buffer)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32I, W, H, 0, gl.GL_RED_INTEGER, gl.GL_INT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_lower)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, W, H, 0, gl.GL_RED, gl.GL_FLOAT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_attach)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, W, H, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    # Bind texture to fbo
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, index_buffer, 0)  # location 1
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_TEXTURE_2D, depth_lower, 0)  # location 1
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, depth_attach, 0)
    gl.glDrawBuffers(2, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1])

    # Check framebuffer status
    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
        log(red('Framebuffer not complete, exiting...'))
        raise RuntimeError('Incomplete framebuffer')

    # Restore the original state
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return index_buffer, depth_lower, depth_attach, fbo


class Gaussian(Mesh):
    def __init__(self,
                 filename: str = 'assets/meshes/zju3dv.npz',

                 gaussian_cfg: dotdict = dotdict(),
                 quad_cfg: dotdict = dotdict(),

                 view_depth: bool = False,  # show depth or show color
                 dpt_cm: str = 'linear',

                 H: int = 1024,
                 W: int = 1024,
                 **kwargs,
                 ):
        # Import Gaussian Model
        from easyvolcap.engine.registry import call_from_cfg
        from easyvolcap.utils.gaussian_utils import GaussianModel

        # Housekeeping
        super().__init__(**kwargs)
        self.name = split(filename)[-1]

        # Init Gaussian related models, for now only the first gaussian model is supported
        if filename.endswith('.npz') or filename.endswith('.pt') or filename.endswith('.pth'):
            # Load from GaussianTSampler
            pretrained, _ = load_pretrained(filename)  # loaded model and updated path (maybe)
            pretrained = pretrained.model
            state_dict = dotdict()
            for k, v in pretrained.items():
                if k.startswith('sampler.pcds.0'):
                    state_dict[k.replace('sampler.pcds.0.', '')] = v

            # Load the parameters into the gaussian model
            self.gaussian_model: GaussianModel = call_from_cfg(GaussianModel, gaussian_cfg)  # init empty gaussian model
            self.gaussian_model.load_state_dict(state_dict)  # load the first gaussian model
            self.gaussian_model.cuda()  # move the parameters to GPU

        elif filename.endswith('.ply'):
            # Load raw GaussianModel
            # pts, rgb, norm, scalars = load_pts(filename)
            self.gaussian_model: GaussianModel = call_from_cfg(GaussianModel, gaussian_cfg)  # init empty gaussian model
            self.gaussian_model.load_ply(filename)  # load the original gaussian model
            self.gaussian_model.cuda()
        else:
            raise NotImplementedError

        # Init rendering quad
        self.quad: Quad = call_from_cfg(Quad, quad_cfg, H=H, W=W)

        # Other configurations
        self.view_depth = view_depth
        self.dpt_cm = dpt_cm
        del self.shade_flat
        del self.point_radius
        del self.render_normal

    # Disabling initialization
    def load_from_file(self, *args, **kwargs):
        pass

    def load_from_data(self, *args, **kwargs):
        pass

    def compile_shaders(self):
        pass

    def update_gl_buffers(self):
        pass

    def resize_textures(self, H: int, W: int):
        self.quad.resize_textures(H, W)

    # The actual rendering function
    @torch.no_grad()
    def render(self, camera: Camera):
        # Perform actual gaussian rendering
        batch = add_batch(to_cuda(camera.to_batch()))
        rgb, acc, dpt = self.gaussian_model.render(batch)

        if self.view_depth:
            rgba = torch.cat([depth_curve_fn(dpt, cm=self.dpt_cm), acc], dim=-1)  # H, W, 4
        else:
            rgba = torch.cat([rgb, acc], dim=-1)  # H, W, 4

        # Copy rendered tensor to screen
        rgba = (rgba.clip(0, 1) * 255).type(torch.uint8).flip(0)  # transform
        self.quad.copy_to_texture(rgba)
        self.quad.draw()

    def render_imgui(mesh, viewer: 'VolumetricVideoViewer', batch: dotdict):
        super().render_imgui(viewer, batch)

        from imgui_bundle import imgui
        from easyvolcap.utils.imgui_utils import push_button_color, pop_button_color

        i = batch.i
        imgui.same_line()
        push_button_color(0x55cc33ff if not mesh.view_depth else 0x8855aaff)
        if imgui.button(f'Color##{i}' if not mesh.view_depth else f' Depth ##{i}'):
            mesh.view_depth = not mesh.view_depth
        pop_button_color()


class PointSplat(Gaussian):
    def __init__(self,
                 filename: str = 'assets/meshes/zju3dv.ply',

                 quad_cfg: dotdict = dotdict(),

                 view_depth: bool = False,  # show depth or show color
                 dpt_cm: str = 'linear',

                 H: int = 1024,
                 W: int = 1024,
                 **kwargs,
                 ):
        # Import Gaussian Model
        from easyvolcap.engine.registry import call_from_cfg
        from easyvolcap.utils.data_utils import load_pts
        from easyvolcap.utils.net_utils import make_buffer
        from easyvolcap.models.samplers.gaussiant_sampler import GaussianTSampler

        # Housekeeping
        super(Gaussian, self).__init__(**kwargs)
        self.name = split(filename)[-1]
        self.render_radius = MethodType(GaussianTSampler.render_radius, self)  # override the method

        # Init PointSplat related models, for now only the first gaussian model is supported
        if filename.endswith('.ply'):
            # Load raw GaussianModel
            pts, rgb, norms, scalars = load_pts(filename)
            occ, rad = scalars.alpha, scalars.radius
            self.pts = make_buffer(torch.from_numpy(pts))  # N, 3
            self.rgb = make_buffer(torch.from_numpy(rgb))  # N, 3
            self.occ = make_buffer(torch.from_numpy(occ))  # N, 1
            self.rad = make_buffer(torch.from_numpy(rad))  # N, 1
        else:
            raise NotImplementedError

        # Init rendering quad
        self.quad: Quad = call_from_cfg(Quad, quad_cfg, H=H, W=W)

        # Other configurations
        self.view_depth = view_depth
        self.dpt_cm = dpt_cm

    # The actual rendering function
    @torch.no_grad()
    def render(self, camera: Camera):
        # Perform actual gaussian rendering
        batch = add_batch(to_cuda(camera.to_batch()))
        sh0 = rgb2sh0(self.rgb)
        xyz = self.pts
        occ = self.occ
        rad = self.rad

        rgb, acc, dpt = self.render_radius(*add_batch([xyz, sh0, rad, occ]), batch)

        if self.view_depth:
            rgba = torch.cat([depth_curve_fn(dpt, cm=self.dpt_cm), acc], dim=-1)  # H, W, 4
        else:
            rgba = torch.cat([rgb, acc], dim=-1)  # H, W, 4

        # Copy rendered tensor to screen
        rgba = (rgba.clip(0, 1) * 255).type(torch.uint8).flip(0)  # transform
        self.quad.copy_to_texture(rgba)
        self.quad.render()


class Splat(Mesh):  # FIXME: Not rendering, need to debug this
    def __init__(self,
                 *args,
                 H: int = 512,
                 W: int = 512,
                 tex_dtype: str = torch.half,

                 pts_per_pix: int = 24,  # render less for the static background since we're only doing a demo
                 blit_last_ratio: float = 0.0,
                 volume_rendering: bool = True,
                 radii_mult_volume: float = 1.00,  # 2 / 3 is the right integration, but will leave holes, 1.0 will make it bloat, 0.85 looks visually better
                 radii_mult_solid: float = 0.85,  # 2 / 3 is the right integration, but will leave holes, 1.0 will make it bloat, 0.85 looks visually better

                 point_smooth: bool = True,
                 alpha_blending: bool = True,
                 **kwargs):
        kwargs = dotdict(kwargs)
        kwargs.vert_sizes = kwargs.get('vert_sizes', [3, 3, 1, 1])
        self.tex_dtype = getattr(torch, tex_dtype) if isinstance(tex_dtype, str) else tex_dtype
        self.gl_tex_dtype = gl.GL_RGBA16F if self.tex_dtype == torch.half else gl.GL_RGBA32F

        super().__init__(*args, **kwargs)
        self.use_gl_program(self.splat_program)

        self.pts_per_pix = pts_per_pix
        self.blit_last_ratio = blit_last_ratio
        self.volume_rendering = volume_rendering
        self.radii_mult_volume = radii_mult_volume
        self.radii_mult_solid = radii_mult_solid

        self.point_smooth = point_smooth
        self.alpha_blending = alpha_blending

        self.max_H, self.max_W = H, W
        self.H, self.W = H, W
        self.init_textures()

    @property
    def verts_data(self):  # a heavy copy operation
        verts = torch.cat([self.verts, self.colors, self.radius, self.alpha], dim=-1).ravel().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=np.float32, order='C')  # this should only be invoked once
        return verts

    def use_gl_program(self, program: shaders.ShaderProgram):
        super().use_gl_program(program)
        # Special controlling variables
        self.uniforms.alpha_blending = gl.glGetUniformLocation(program, f'alpha_blending')
        self.uniforms.point_smooth = gl.glGetUniformLocation(program, f'point_smooth')
        self.uniforms.radii_mult = gl.glGetUniformLocation(program, f'radii_mult')

        # Special rendering variables
        self.uniforms.pass_index = gl.glGetUniformLocation(program, f'pass_index')
        self.uniforms.read_color = gl.glGetUniformLocation(program, f'read_color')
        self.uniforms.read_upper = gl.glGetUniformLocation(program, f'read_upper')
        self.uniforms.read_lower = gl.glGetUniformLocation(program, f'read_lower')
        gl.glUniform1i(self.uniforms.read_color, 0)
        gl.glUniform1i(self.uniforms.read_upper, 1)
        gl.glUniform1i(self.uniforms.read_lower, 2)

    def compile_shaders(self):
        try:
            self.splat_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('splat.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('splat.frag'), gl.GL_FRAGMENT_SHADER)
            )
            self.usplat_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('usplat.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('usplat.frag'), gl.GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    def rasterize(self, camera: Camera = None, length: int = None):
        if self.volume_rendering:
            return self.rasterize_volume(camera, length)
        else:
            return self.rasterize_solid(camera, length)

    def rasterize_volume(self, camera: Camera = None, length: int = None):  # some implementation requires no uploading of camera
        """
        Let's try to analyze what's happening here

        We want to:
            1. Render the front-most color to color buffer
            2. UNUSED: Render the front-most depth + some large margin to a depth upper limit buffer 
            3. Render the front-most depth + some small margin to a depth lower limit buffer
            4. Switch between the render target and sampling target
            5. Use the previous rendered color, depth upper limit and lower limit as textures
            6. When current depth is smaller than the lower limit, we've already rendered this in the first pass, discard
            7. UNUSED: When current depth is larger than the upper limit, it will probabily not contribute much to final results, discard
            8. UNUSED: When the accumulated opacity reaches almost 1, subsequent rendering would not have much effect, return directly
            9. When the point coordinates falls out of bound of the current sphere, dicard (this could be optimized with finutining in rectangle)
            10. Finally, try to render the final color using the volume rendering equation (by accumulating alpha values from front to back)

        Required cleanup checklist:
            1. Before rendering the first pass, we need to clear the color and depth texture, this is not done, need to check multi-frame accumulation on this
            2. Before rendering next pass, it's also recommended to blit color and depth values from previous pass to avoid assign them in the shader
        """

        front_fbo, front_color, front_upper, front_lower = self.read_fbo, self.read_color, self.read_upper, self.read_lower
        back_fbo, back_color, back_upper, back_lower = self.write_fbo, self.write_color, self.write_upper, self.write_lower

        # Only clear the output once
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, front_fbo)  # for offscreen rendering to textures
        gl.glClearBufferfv(gl.GL_COLOR, 0, [0.0, 0.0, 0.0, 0.0])
        # gl.glClearBufferfv(gl.GL_COLOR, 1, [1e9])
        gl.glClearBufferfv(gl.GL_COLOR, 2, [0.0, 0.0, 0.0, 0.0])
        gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

        # Only clear the output once
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures
        gl.glClearBufferfv(gl.GL_COLOR, 0, [0.0, 0.0, 0.0, 0.0])
        # gl.glClearBufferfv(gl.GL_COLOR, 1, [1e9])
        gl.glClearBufferfv(gl.GL_COLOR, 2, [0.0, 0.0, 0.0, 0.0])
        gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

        # Prepare for the actual rendering, previous operations could rebind the vertex array
        self.use_gl_program(self.splat_program)  # TODO: Implement this with a mapping and a lazy modification
        self.upload_gl_uniforms(camera)
        gl.glBindVertexArray(self.vao)

        # The actual multi pass rendering process happens here
        for pass_index in range(self.pts_per_pix):
            # Swap buffers to render the next pass
            front_fbo, front_color, front_upper, front_lower, back_fbo, back_color, back_upper, back_lower = \
                back_fbo, back_color, back_upper, back_lower, front_fbo, front_color, front_upper, front_lower

            # Bind the read texture and bind the write render frame buffer
            gl.glBindTextures(0, 3, [front_color, front_upper, front_lower])

            # Move content from write_fbo to screen fbo
            if pass_index > self.pts_per_pix * self.blit_last_ratio:  # no blitting almost has no effect on the rendering
                gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, front_fbo)
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, back_fbo)
                for i in range(3):
                    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
                    gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
                    gl.glBlitFramebuffer(0, 0, self.W, self.H, 0, 0, self.W, self.H, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
                gl.glDrawBuffers(3, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2])

            # Clear depth buffer for depth testing
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures
            gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing
            gl.glUniform1i(self.uniforms.pass_index, pass_index)  # pass index

            # The actual drawing pass with render things out to the write_fbo
            gl.glDrawArrays(gl.GL_POINTS, 0, length if length is not None else len(self.verts))  # number of vertices

        # Restore states of things
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindVertexArray(0)
        return back_fbo

    def upload_gl_uniforms(self, camera: Camera):
        super().upload_gl_uniforms(camera)
        gl.glUniform1i(self.uniforms.point_smooth, self.point_smooth)
        gl.glUniform1i(self.uniforms.alpha_blending, self.alpha_blending)

        if self.volume_rendering:
            gl.glUniform1f(self.uniforms.radii_mult, self.radii_mult_volume)  # radii mult
        else:
            gl.glUniform1f(self.uniforms.radii_mult, self.radii_mult_solid)  # radii mult

    def rasterize_solid(self, camera: Camera = None, length: int = None):
        # Only clear the output once
        back_fbo = self.write_fbo
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures
        gl.glClearBufferfv(gl.GL_COLOR, 0, [0.0, 0.0, 0.0, 0.0])  # color
        # gl.glClearBufferfv(gl.GL_COLOR, 1, [0.0]) # depth upper
        gl.glClearBufferfv(gl.GL_COLOR, 2, [0.0, 0.0, 0.0, 0.0])  # depth lower
        gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

        # Prepare for the actual rendering, previous operations could rebind the vertex array
        self.use_gl_program(self.usplat_program)
        self.upload_gl_uniforms(camera)
        gl.glUniform1i(self.uniforms.pass_index, 0)  # pass index
        gl.glBindVertexArray(self.vao)

        # The actual drawing pass with render things out to the write_fbo
        gl.glDrawArrays(gl.GL_POINTS, 0, length if length is not None else len(self.verts))  # number of vertices

        # Restore states of things
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindVertexArray(0)
        return back_fbo

    def show(self, back_fbo: int):
        # Move content from write_fbo to screen fbo
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, back_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)  # render the final content onto screen
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBlitFramebuffer(0, 0, self.W, self.H, 0, 0, self.W, self.H, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def render(self, camera):
        if not self.visible: return
        self.show(self.rasterize(camera))

    def resize_textures(self, H: int, W: int):  # analogy to update_gl_buffers
        self.H, self.W = H, W
        if self.H > self.max_H or self.W > self.max_W:  # max got updated
            self.max_H, self.max_W = max(int(self.H * 1.05), self.max_H), max(int(self.W * 1.05), self.max_W)
            self.init_textures()

    def init_textures(self):
        if hasattr(self, 'write_fbo'):
            gl.glDeleteFramebuffers(2, [self.write_fbo, self.read_fbo])
            gl.glDeleteTextures(8, [self.write_color, self.write_upper, self.write_lower, self.write_attach, self.read_color, self.read_upper, self.read_lower, self.read_attach])
        self.write_color, self.write_upper, self.write_lower, self.write_attach, self.write_fbo = hardware_rendering_framebuffer(self.max_H, self.max_W, self.gl_tex_dtype)
        self.read_color, self.read_upper, self.read_lower, self.read_attach, self.read_fbo = hardware_rendering_framebuffer(self.max_H, self.max_W, self.gl_tex_dtype)
        log(f'Created texture of h, w: {self.max_H}, {self.max_W}')


class HardwareRendering(Splat):
    def __init__(self,
                 dtype=torch.half,
                 **kwargs,
                 ):
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.gl_dtype = gl.GL_HALF_FLOAT if self.dtype == torch.half else gl.GL_FLOAT
        kwargs = dotdict(kwargs)
        kwargs.blit_last_ratio = kwargs.get('blit_last_ratio', 0.90)
        kwargs.vert_sizes = kwargs.get('vert_sizes', [3, 3, 1, 1])
        super().__init__(**kwargs)  # verts, color, radius, alpha

    @property
    def verts_data(self):  # a heavy copy operation
        verts = torch.cat([self.verts, self.colors, self.radius, self.alpha], dim=-1).ravel().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=torch_dtype_to_numpy_dtype(self.dtype), order='C')  # this should only be invoked once
        return verts

    def init_gl_buffers(self, v: int = 0, f: int = 0):
        from cuda import cudart
        if hasattr(self, 'cu_vbo'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_vbo))

        super().init_gl_buffers(v, f)

        # Register vertex buffer obejct
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        try:
            self.cu_vbo = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterBuffer(self.vbo, flags))
        except RuntimeError as e:
            log(red(f'Your system does not support CUDA-GL interop, please use pytorch3d\'s implementation instead'))
            log(red(f'This can be done by specifying {blue("model_cfg.sampler_cfg.use_cudagl=False model_cfg.sampler_cfg.use_diffgl=False")} at the end of your command'))
            log(red(f'Note that this implementation is extremely slow, we recommend running on a native system that support the interop'))
            # raise RuntimeError(str(e) + ": This unrecoverable, please read the error message above")
            raise e

    def init_textures(self):
        from cuda import cudart
        if hasattr(self, 'cu_read_color'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_read_color))
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_write_color))
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_read_lower))
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_write_lower))

        super().init_textures()

        # Register image to read from
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
        self.cu_read_color = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.read_color, gl.GL_TEXTURE_2D, flags))
        self.cu_write_color = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.write_color, gl.GL_TEXTURE_2D, flags))
        self.cu_read_lower = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.read_lower, gl.GL_TEXTURE_2D, flags))
        self.cu_write_lower = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.write_lower, gl.GL_TEXTURE_2D, flags))

    def forward(self, xyz: torch.Tensor, rgb: torch.Tensor, rad: torch.Tensor, occ: torch.Tensor, batch: dotdict):
        """
        Renders a 3D point cloud using OpenGL and returns the rendered RGB image, accumulated alpha image, and depth map.

        Args:
            xyz (torch.Tensor): A tensor of shape (B, N, 3) containing the 3D coordinates of the points.
            rgb (torch.Tensor): A tensor of shape (B, N, 3) containing the RGB color values of the points.
            rad (torch.Tensor): A tensor of shape (B, N, 1) containing the radii of the points.
            batch (dotdict): A dictionary containing the camera parameters and other metadata for the batch.

        Returns:
            A tuple containing the rendered RGB image, accumulated alpha image, and depth map, all as torch.Tensors.
            The RGB image has shape (1, H, W, 3), the alpha image has shape (1, H, W, 1), and the depth map has shape (1, H, W, 1).

        The method first resizes the OpenGL texture to match the height and width of the output image. It then sets the OpenGL viewport and scissor to only render in the region of the viewport specified by the output image size.

        It concatenates the `xyz`, `rgb`, and `rad` tensors along the last dimension and flattens the result into a 1D tensor.

        The method then uploads the input data to OpenGL for rendering and performs depth peeling using OpenGL. The method uploads the camera parameters to OpenGL and renders the point cloud, saving the output buffer to the `back_fbo` attribute of the class.

        Finally, the method copies the rendered image and depth back to the CPU as torch.Tensors and reshapes them to match the output image size. The RGB image is returned with shape (1, H, W, 3), the accumulated alpha image is returned with shape (1, H, W, 1), and the depth map is returned with shape (1, H, W, 1).
        """
        from cuda import cudart
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

        # !: BATCH
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        self.resize_textures(H, W)  # maybe resize the texture
        self.resize_buffers(xyz.shape[1])  # maybe resize the buffer
        _, _, old_W, old_H = gl.glGetIntegerv(gl.GL_VIEWPORT)
        gl.glViewport(0, 0, W, H)
        gl.glScissor(0, 0, W, H)  # only render in this small region of the viewport

        # Prepare for input data
        data = torch.cat([xyz, rgb, rad, occ], dim=-1).type(self.dtype).ravel()

        # Upload to opengl for rendering
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.cu_vbo, torch.cuda.current_stream().cuda_stream))
        cu_vbo_ptr, cu_vbo_size = CHECK_CUDART_ERROR(cudart.cudaGraphicsResourceGetMappedPointer(self.cu_vbo))
        assert cu_vbo_size >= data.numel() * data.element_size(), f'PyTorch(CUDA) and OpenGL vertex buffer size mismatch ({data.numel() * data.element_size()} v.s. {cu_vbo_size}), CUDA side should be less than or equal to the OpenGL side'
        CHECK_CUDART_ERROR(cudart.cudaMemcpyAsync(cu_vbo_ptr,
                                                  data.data_ptr(),
                                                  data.numel() * data.element_size(),
                                                  kind,
                                                  torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.cu_vbo, torch.cuda.current_stream().cuda_stream))

        # Perform rasterization (depth peeling using OpenGL)
        if 'meta_stream' in batch.meta: batch.meta.meta_stream.synchronize()  # wait for gpu -> cpu copy to finish
        back_fbo = self.rasterize(Camera(batch=batch.meta), xyz.shape[-2])  # will upload and render, save output buffer to back_fbo

        # Copy rendered image and depth back as tensor
        cu_tex = self.cu_write_color if back_fbo == self.write_fbo else self.cu_read_color  # double buffered depth peeling
        cu_dpt = self.cu_write_lower if back_fbo == self.write_fbo else self.cu_read_lower  # double buffered depth peeling

        # Prepare the output # !: BATCH
        rgb_map = torch.empty((H, W, 4), dtype=self.tex_dtype, device='cuda')  # to hold the data from opengl
        dpt_map = torch.empty((H, W, 1), dtype=torch.float, device='cuda')  # to hold the data from opengl

        # The resources in resources may be accessed by CUDA until they are unmapped.
        # The graphics API from which resources were registered should not access any resources while they are mapped by CUDA.
        # If an application does so, the results are undefined.
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, cu_tex, torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, cu_dpt, torch.cuda.current_stream().cuda_stream))
        cu_tex_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(cu_tex, 0, 0))
        cu_dpt_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(cu_dpt, 0, 0))
        CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(rgb_map.data_ptr(),  # dst
                                                             W * 4 * rgb_map.element_size(),  # dpitch
                                                             cu_tex_arr,  # src
                                                             0,  # wOffset
                                                             0,  # hOffset
                                                             W * 4 * rgb_map.element_size(),  # width Width of matrix transfer (columns in bytes)
                                                             H,  # height
                                                             kind,  # kind
                                                             torch.cuda.current_stream().cuda_stream))  # stream
        CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(dpt_map.data_ptr(),
                                                             W * 1 * dpt_map.element_size(),
                                                             cu_dpt_arr,
                                                             0,
                                                             0,
                                                             W * 1 * dpt_map.element_size(),
                                                             H,
                                                             kind,
                                                             torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, cu_tex, torch.cuda.current_stream().cuda_stream))  # MARK: SYNC
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, cu_dpt, torch.cuda.current_stream().cuda_stream))  # MARK: SYNC

        # Ouput reshaping
        rgb_map, dpt_map = rgb_map[None].flip(1), dpt_map[None].flip(1)
        rgb_map, acc_map = rgb_map[..., :3], rgb_map[..., 3:]
        dpt_map = torch.where(dpt_map == 0, dpt_map.max(), dpt_map)

        # Some house keepings
        gl.glViewport(0, 0, old_W, old_H)
        gl.glScissor(0, 0, old_W, old_H)
        return rgb_map, acc_map, dpt_map


class HardwarePeeling(Splat):
    def __init__(self,
                 dtype=torch.float,
                 **kwargs):
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.gl_dtype = gl.GL_HALF_FLOAT if self.dtype == torch.half else gl.GL_FLOAT
        super().__init__(**kwargs,
                         blit_last_ratio=-10.0,
                         vert_sizes=[3, 1],
                         )  # verts, radius, index
        # from pytorch3d.renderer import AlphaCompositor
        # self.compositor = AlphaCompositor()  # this the key to convergence, this is differentiable

    @property
    def verts_data(self):  # a heavy copy operation
        verts = torch.cat([self.verts, self.radius], dim=-1).ravel().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=torch_dtype_to_numpy_dtype(self.dtype), order='C')  # this should only be invoked once
        return verts

    def init_gl_buffers(self, v: int = 0, f: int = 0):
        from cuda import cudart
        if hasattr(self, 'cu_vbo'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_vbo))

        super().init_gl_buffers(v, f)

        # Register vertex buffer obejct
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        self.cu_vbo = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterBuffer(self.vbo, flags))\


    def use_gl_program(self, program):
        super().use_gl_program(program)

        gl.glUseProgram(self.splat_program)  # use a different program
        self.uniforms.read_index = gl.glGetUniformLocation(program, f'read_index')
        self.uniforms.read_lower = gl.glGetUniformLocation(program, f'read_lower')
        gl.glUniform1i(self.uniforms.read_index, 0)
        gl.glUniform1i(self.uniforms.read_lower, 1)

    def upload_gl_uniforms(self, camera: Camera):
        super().upload_gl_uniforms(camera)

    def compile_shaders(self):
        try:
            self.splat_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('idx_splat.vert'), gl.GL_VERTEX_SHADER),  # use the pass through quad shader
                shaders.compileShader(load_shader_source('idx_splat.frag'), gl.GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    def init_textures(self):
        from cuda import cudart
        if hasattr(self, 'cu_read_index'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_read_index))
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_write_index))
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_read_lower))
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_write_lower))

        if hasattr(self, 'write_fbo'):
            gl.glDeleteFramebuffers(2, [self.write_fbo, self.read_fbo])
            gl.glDeleteTextures(6, [self.write_index, self.write_lower, self.write_attach, self.read_index, self.read_lower, self.read_attach])

        self.write_index, self.write_lower, self.write_attach, self.write_fbo = hareward_peeling_framebuffer(self.max_H, self.max_W)
        self.read_index, self.read_lower, self.read_attach, self.read_fbo = hareward_peeling_framebuffer(self.max_H, self.max_W)

        # Register image to read from
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
        self.cu_read_index = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.read_index, gl.GL_TEXTURE_2D, flags))
        self.cu_write_index = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.write_index, gl.GL_TEXTURE_2D, flags))
        self.cu_read_lower = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.read_lower, gl.GL_TEXTURE_2D, flags))
        self.cu_write_lower = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.write_lower, gl.GL_TEXTURE_2D, flags))

        log(f'Created texture of h, w: {self.max_H}, {self.max_W}')

    def rasterize_generator(self, camera: Camera = None, length: int = None):  # some implementation requires no uploading of camera
        front_fbo, front_index, front_lower = self.read_fbo, self.read_index, self.read_lower
        back_fbo, back_index, back_lower = self.write_fbo, self.write_index, self.write_lower

        # Only clear the output once
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, front_fbo)  # for offscreen rendering to textures
        gl.glClearBufferiv(gl.GL_COLOR, 0, [-1])
        gl.glClearBufferfv(gl.GL_COLOR, 1, [0.0])
        gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

        # Only clear the output once
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures
        gl.glClearBufferiv(gl.GL_COLOR, 0, [-1])
        gl.glClearBufferfv(gl.GL_COLOR, 1, [0.0])
        gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

        # Prepare for the actual rendering, previous operations could rebind the vertex array
        self.use_gl_program(self.splat_program)
        self.upload_gl_uniforms(camera)
        gl.glBindVertexArray(self.vao)

        # The actual multi pass rendering process happens here
        for pass_index in range(self.pts_per_pix):
            # Swap buffers to render the next pass
            front_fbo, front_index, front_lower, back_fbo, back_index, back_lower = \
                back_fbo, back_index, back_lower, front_fbo, front_index, front_lower

            # Bind the read texture and bind the write render frame buffer
            gl.glBindTextures(0, 2, [front_index, front_lower])

            # Move content from write_fbo to screen fbo
            if pass_index > self.pts_per_pix * self.blit_last_ratio:  # no blitting almost has no effect on the rendering
                gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, front_fbo)
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, back_fbo)
                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + 1)
                gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0 + 1)
                gl.glBlitFramebuffer(0, 0, self.W, self.H, 0, 0, self.W, self.H, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures
                gl.glDrawBuffers(2, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1])
            else:
                # Only clear the output once
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures

            # Clear depth buffer for depth testing
            gl.glClearBufferiv(gl.GL_COLOR, 0, [-1])  # clear the indices buffer for later rendering and retrieving
            gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

            # The actual drawing pass with render things out to the write_fbo
            gl.glDrawArrays(gl.GL_POINTS, 0, length if length is not None else len(self.verts))  # number of vertices
            yield back_fbo  # give the CUDA end a chance to read from this frame buffer after rendering

        # Restore states of things
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindVertexArray(0)
        return

    def forward(self,
                xyz: torch.Tensor, rgb: torch.Tensor, rad: torch.Tensor, occ: torch.Tensor,
                batch: dotdict,
                return_frags: bool = False,
                return_full: bool = False,
                ):
        """
        Get all indices from the depth peeling passes
        Compute the vertex weight here in torch(cuda)
        Use the indices to pass through a compositor
        The backward pass should only be valid on the torch side, and it should've been enough

        TODO: This function is too memory intensive
        TODO: Performing IBR is too memory intensive
        """

        # This the slow part, but not differentiable
        idx, _, _ = self.forward_idx(xyz, rad, batch)  # B, H, W, K
        msk = idx != -1  # B, H, W, K
        idx = torch.where(msk, idx, 0).long()

        # Sample things needed for computing screen space weight
        H, W, K, R, T, C = get_opencv_camera_params(batch)
        K, R, T, C = K.to(xyz.dtype), R.to(xyz.dtype), T.to(xyz.dtype), C.to(xyz.dtype)
        pix_xyz = (xyz @ R.mT + T.mT) @ K.mT  # B, P, 3
        pix_xyz_xy = pix_xyz[..., :-1] / (pix_xyz[..., -1:] + 1e-10)
        pix_rad = abs(K[..., 1, 1][..., None] * rad[..., 0] / (pix_xyz[..., -1] + 1e-10))  # z: B, 1 * B, N, world space radius

        mean_xy = multi_gather(pix_xyz_xy, idx.view(idx.shape[0], -1), dim=-2).view(*idx.shape, 2)  # B, HWK, 2 -> B, H, W, K, 2
        xy = create_meshgrid(H, W, idx.device, dtype=xyz.dtype).flip(-1)[None].expand(idx.shape[0], H, W, 2)  # create screen space xy (opencv)
        dists = (xy[..., None, :] - mean_xy).pow(2).sum(-1)  # B, H, W, K

        # Point values
        dpt = (xyz - C.mT).norm(dim=-1, keepdim=True)  # B, N, 1
        pix_occ = multi_gather(occ, idx.view(idx.shape[0], -1), dim=-2).view(*idx.shape)
        pix_rad = multi_gather(pix_rad, idx.view(idx.shape[0], -1), dim=-1).view(*idx.shape)  # -> B, H, W, K
        pix_occ = pix_occ * (1 - dists / (pix_rad * pix_rad + 1e-10))  # B, H, W, K
        pix_occ = pix_occ.clip(0, 1)
        pix_occ = torch.where(msk, pix_occ, 0)

        if return_frags:
            return idx, pix_occ  # B, H, W, K

        # The actual computation
        rgb = torch.cat([rgb, occ, dpt], dim=-1)  # B, N, 3 + C
        pix_rgb = multi_gather(rgb, idx.view(idx.shape[0], -1), dim=-2).view(*idx.shape, rgb.shape[-1])  # B, H, W, K, -1
        _, rgb, _ = volume_rendering(pix_rgb, pix_occ[..., None])  # B, H, W, -1

        rgb, acc, dpt = rgb[..., :-2], rgb[..., -2:-1], rgb[..., -1:]
        dpt = dpt + (1 - acc) * dpt.max()  # only for the looks (rendered depth are already premultiplied)

        if return_full:
            return rgb, acc, dpt, idx, pix_occ
        else:
            return rgb, acc, dpt

    def forward_idx(self, xyz: torch.Tensor, rad: torch.Tensor, batch: dotdict):

        from cuda import cudart
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

        # !: BATCH
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        self.resize_textures(H, W)  # maybe resize the texture
        self.resize_buffers(xyz.shape[1])  # maybe resize the buffer
        _, _, old_W, old_H = gl.glGetIntegerv(gl.GL_VIEWPORT)
        gl.glViewport(0, 0, W, H)
        gl.glScissor(0, 0, W, H)  # only render in this small region of the viewport

        # Prepare for input data
        data = torch.cat([xyz, rad], dim=-1).type(self.dtype).ravel()

        # Upload to opengl for rendering
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.cu_vbo, torch.cuda.current_stream().cuda_stream))
        cu_vbo_ptr, cu_vbo_size = CHECK_CUDART_ERROR(cudart.cudaGraphicsResourceGetMappedPointer(self.cu_vbo))
        assert cu_vbo_size >= data.numel() * data.element_size(), f'PyTorch(CUDA) and OpenGL vertex buffer size mismatch ({data.numel() * data.element_size()} v.s. {cu_vbo_size}), CUDA side should be less than or equal to the OpenGL side'
        CHECK_CUDART_ERROR(cudart.cudaMemcpyAsync(cu_vbo_ptr,
                                                  data.data_ptr(),
                                                  data.numel() * data.element_size(),
                                                  kind,
                                                  torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.cu_vbo, torch.cuda.current_stream().cuda_stream))

        # Perform rasterization (depth peeling using OpenGL)
        if 'meta_stream' in batch.meta: batch.meta.meta_stream.synchronize()  # wait for gpu -> cpu copy to finish
        # FIXME: Strange bug occurs if batch parameter is passed in directly for the construction of Camera(batch=batch.meta)
        gen = self.rasterize_generator(Camera(batch=batch.meta), xyz.shape[-2])  # will upload and render, save output buffer to back_fbo

        ind_maps = []
        dpt_maps = []
        acc_maps = []
        for back_fbo in gen:
            # Copy rendered image and depth back as tensor
            cu_tex = self.cu_write_index if back_fbo == self.write_fbo else self.cu_read_index  # double buffered depth peeling
            cu_dpt = self.cu_write_lower if back_fbo == self.write_fbo else self.cu_read_lower  # double buffered depth peeling

            # Prepare the output # !: BATCH
            ind_map = torch.empty((H, W, 1), dtype=torch.int, device='cuda')  # to hold the data from opengl
            dpt_map = torch.empty((H, W, 1), dtype=torch.float, device='cuda')  # to hold the data from opengl

            # The resources in resources may be accessed by CUDA until they are unmapped.
            # The graphics API from which resources were registered should not access any resources while they are mapped by CUDA.
            # If an application does so, the results are undefined.
            CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, cu_tex, torch.cuda.current_stream().cuda_stream))
            CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, cu_dpt, torch.cuda.current_stream().cuda_stream))
            cu_tex_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(cu_tex, 0, 0))
            cu_dpt_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(cu_dpt, 0, 0))
            CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(ind_map.data_ptr(),  # dst
                                                                 W * ind_map.shape[-1] * ind_map.element_size(),  # dpitch
                                                                 cu_tex_arr,  # src
                                                                 0,  # wOffset
                                                                 0,  # hOffset
                                                                 W * ind_map.shape[-1] * ind_map.element_size(),  # width Width of matrix transfer (columns in bytes)
                                                                 H,  # height
                                                                 kind,  # kind
                                                                 torch.cuda.current_stream().cuda_stream))  # stream
            CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(dpt_map.data_ptr(),
                                                                 W * dpt_map.shape[-1] * dpt_map.element_size(),
                                                                 cu_dpt_arr,
                                                                 0,
                                                                 0,
                                                                 W * dpt_map.shape[-1] * dpt_map.element_size(),
                                                                 H,
                                                                 kind,
                                                                 torch.cuda.current_stream().cuda_stream))
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, cu_tex, torch.cuda.current_stream().cuda_stream))  # MARK: SYNC
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, cu_dpt, torch.cuda.current_stream().cuda_stream))  # MARK: SYNC

            # Ouput reshaping
            ind_map, dpt_map = ind_map[None].flip(1), dpt_map[None].flip(1)
            acc_map = ind_map != -1
            dpt_map = torch.where(dpt_map == 0, dpt_map.max(), dpt_map)

            ind_maps.append(ind_map)
            acc_maps.append(acc_map)
            dpt_maps.append(dpt_map)

        ind_map = torch.cat(ind_maps, dim=-1)  # B, H, W, K
        acc_map = torch.cat(acc_maps, dim=-1)  # B, H, W, K
        dpt_map = torch.cat(dpt_maps, dim=-1)  # B, H, W, K

        # Some house keepings
        gl.glViewport(0, 0, old_W, old_H)
        gl.glScissor(0, 0, old_W, old_H)
        return ind_map, acc_map, dpt_map
