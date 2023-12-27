"""Headless GPU-accelerated OpenGL context creation on Google Colaboratory.

Typical usage:

    # Optional PyOpenGL configuratiopn can be done here.
    # import OpenGL
    # OpenGL.ERROR_CHECKING = True

    # 'glcontext' must be imported before any OpenGL.* API.
    from lucid.misc.gl.glcontext import create_opengl_context

    # Now it's safe to import OpenGL and EGL functions
    import OpenGL.GL as gl

    # create_opengl_context() creates a GL context that is attached to an
    # offscreen surface of the specified size. Note that rendering to buffers
    # of other sizes and formats is still possible with OpenGL Framebuffers.
    #
    # Users are expected to directly use the EGL API in case more advanced
    # context management is required.
    create_opengl_context()

    # OpenGL context is available here.

Please add this to your system:
/usr/share/glvnd/egl_vendor.d/10_nvidia.json

{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
"""

from __future__ import print_function, annotations

import os
import ctypes
from ctypes import pointer, util

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    CUDA_VISIBLE_DEVICES = list(map(int, [i for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if i]))  # remove ''
    from easyvolcap.utils.dist_utils import get_rank
    # os.environ['EGL_DEVICE_ID'] = str(CUDA_VISIBLE_DEVICES[get_rank()])  # TODO: debug this and figure out what `torchrun` does behind the curtain
    os.environ['EGL_DEVICE_ID'] = str(get_rank())


# # [1] https://devblogs.nvidia.com/egl-eye-opengl-visualization-without-x-server/
# # [2] https://devblogs.nvidia.com/linking-opengl-server-side-rendering/
# # [3] https://bugs.python.org/issue9998
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# _find_library_old = ctypes.util.find_library
# try:
#     def _find_library_new(name):
#         return {
#             'GL': 'libOpenGL.so',
#             'EGL': 'libEGL.so',
#         }.get(name, _find_library_old(name))
#     ctypes.util.find_library = _find_library_new
#     import OpenGL.GL as gl
#     import OpenGL.EGL as egl
# except:
#     raise ImportError('Unable to load OpenGL EGL libraries. Make sure you use GPU-enabled backend. Press "Runtime->Change runtime type" and set "Hardware accelerator" to GPU.')
# finally:
#     ctypes.util.find_library = _find_library_old

# fmt: off
"""Extends OpenGL.EGL with definitions necessary for headless rendering."""
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from OpenGL.platform import ctypesloader  # pylint: disable=g-bad-import-order
try:
    # Nvidia driver seems to need libOpenGL.so (as opposed to libGL.so)
    # for multithreading to work properly. We load this in before everything else.
    ctypesloader.loadLibrary(ctypes.cdll, 'OpenGL', mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

# pylint: disable=g-import-not-at-top
from OpenGL import EGL as egl
from OpenGL import GL as gl
from OpenGL import error
# fmt: on

# From the EGL_EXT_device_enumeration extension.
PFNEGLQUERYDEVICESEXTPROC = ctypes.CFUNCTYPE(
    egl.EGLBoolean,
    egl.EGLint,
    ctypes.POINTER(egl.EGLDeviceEXT),
    ctypes.POINTER(egl.EGLint),
)
try:
    _eglQueryDevicesEXT = PFNEGLQUERYDEVICESEXTPROC(  # pylint: disable=invalid-name
        egl.eglGetProcAddress('eglQueryDevicesEXT'))
except TypeError as e:
    raise ImportError('eglQueryDevicesEXT is not available.') from e

EGL_CUDA_DEVICE_NV = 0x323A
PFNEGLQUERYDEVICESATTRIBEXTPROC = ctypes.CFUNCTYPE(
    egl.EGLBoolean,
    egl.EGLDeviceEXT,
    egl.EGLint,
    ctypes.POINTER(ctypes.c_longlong),
)
try:
    eglQueryDeviceAttribEXT = PFNEGLQUERYDEVICESATTRIBEXTPROC(
        egl.eglGetProcAddress('eglQueryDeviceAttribEXT')
    )
except TypeError as e:
    raise ImportError('eglQueryDeviceAttribEXT is not available.') from e

# From the EGL_EXT_platform_device extension.
EGL_PLATFORM_DEVICE_EXT = 0x313F
PFNEGLGETPLATFORMDISPLAYEXTPROC = ctypes.CFUNCTYPE(
    egl.EGLDisplay, egl.EGLenum, ctypes.c_void_p, ctypes.POINTER(egl.EGLint))
try:
    eglGetPlatformDisplayEXT = PFNEGLGETPLATFORMDISPLAYEXTPROC(  # pylint: disable=invalid-name
        egl.eglGetProcAddress('eglGetPlatformDisplayEXT'))
except TypeError as e:
    raise ImportError('eglGetPlatformDisplayEXT is not available.') from e


# Wrap raw _eglQueryDevicesEXT function into something more Pythonic.
def eglQueryDevicesEXT(max_devices=10):  # pylint: disable=invalid-name
    devices = (egl.EGLDeviceEXT * max_devices)()
    num_devices = egl.EGLint()
    success = _eglQueryDevicesEXT(max_devices, devices, num_devices)
    if success == egl.EGL_TRUE:
        return [devices[i] for i in range(num_devices.value)]
    else:
        from OpenGL import error
        raise error.GLError(err=egl.eglGetError(),
                            baseOperation=eglQueryDevicesEXT,
                            result=success)


def create_initialized_egl_device_display():
    """Creates an initialized EGL display directly on a device."""
    all_devices = eglQueryDevicesEXT()
    selected_device = os.environ.get('EGL_DEVICE_ID', None)
    if selected_device is None:
        candidates = all_devices
    else:
        selected_device = int(selected_device)
        for device_idx, device in enumerate(all_devices):
            value = ctypes.c_longlong(-1)
            success = eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, ctypes.byref(value))
            if success == egl.EGL_TRUE and value.value == selected_device:
                break
        if not 0 <= device_idx < len(all_devices):
            raise RuntimeError(
                f'The EGL_DEVICE_ID environment variable must be an integer '
                f'between 0 and {len(all_devices)-1} (inclusive), got {device_idx}.')
        candidates = all_devices[device_idx:device_idx + 1]
    for device in candidates:
        display = eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT, device, None)
        if display != egl.EGL_NO_DISPLAY and egl.eglGetError() == egl.EGL_SUCCESS:
            from OpenGL import error
            # `eglInitialize` may or may not raise an exception on failure depending
            # on how PyOpenGL is configured. We therefore catch a `GLError` and also
            # manually check the output of `eglGetError()` here.
            try:
                initialized = egl.eglInitialize(display, None, None)
            except error.GLError:
                pass
            else:
                if initialized == egl.EGL_TRUE and egl.eglGetError() == egl.EGL_SUCCESS:
                    return display
    return egl.EGL_NO_DISPLAY


def create_opengl_context():
    """Create offscreen OpenGL context and make it current.
    Users are expected to directly use EGL API in case more advanced
    context management is required.
    """
    egl_display = create_initialized_egl_device_display()
    if egl_display == egl.EGL_NO_DISPLAY:
        raise ImportError(
            'Cannot initialize a EGL device display. This likely means that your EGL '
            'driver does not support the PLATFORM_DEVICE extension, which is '
            'required for creating a headless rendering context.')

    config_attribs = [
        egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT,
        egl.EGL_BLUE_SIZE, 8,
        egl.EGL_GREEN_SIZE, 8,
        egl.EGL_RED_SIZE, 8,
        egl.EGL_ALPHA_SIZE, 8,
        egl.EGL_DEPTH_SIZE, 24,
        egl.EGL_STENCIL_SIZE, 8,
        egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT, egl.EGL_NONE
    ]
    config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)
    num_configs = egl.EGLint()
    egl_cfg = egl.EGLConfig()
    egl.eglChooseConfig(egl_display, config_attribs, pointer(egl_cfg), 1, pointer(num_configs))

    egl.eglBindAPI(egl.EGL_OPENGL_API)
    egl_context = egl.eglCreateContext(egl_display, egl_cfg, egl.EGL_NO_CONTEXT, None)
    egl.eglMakeCurrent(egl_display, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl_context)
    return egl_context


class eglContextManager:
    # Manages the creation and destruction of an EGL context
    # Will resize if the size of the window changes
    # Will also manage gl.Viewport to render different parts of the screen
    # Only resize the underlying egl ctx when exceeding current size
    def __init__(self, W=1920, H=1080) -> None:
        self.H, self.W = H, W
        self.max_H, self.max_W = H, W  # always create at first
        self.eglctx = create_opengl_context()
        self.create_fbo_with_rbos(W, H)
        self.resize(W, H)  # maybe create new framebuffer

    def create_fbo_with_rbos(self, W: int, H: int):
        if hasattr(self, 'fbo'):
            gl.glDeleteFramebuffers(1, [self.fbo])
            gl.glDeleteRenderbuffers(6, [self.rbo0, self.rbo1, self.rbo2, self.rbo3, self.rbo4, self.rbo_dpt])

        # Add new buffer
        self.fbo = gl.glGenFramebuffers(1)
        self.rbo0, self.rbo1, self.rbo2, self.rbo3, self.rbo4, self.rbo_dpt = gl.glGenRenderbuffers(6)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo0)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, W, H)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo1)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, W, H)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo2)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, W, H)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo3)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, W, H)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo4)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, W, H)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rbo_dpt)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, W, H)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, self.rbo0)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_RENDERBUFFER, self.rbo1)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_RENDERBUFFER, self.rbo2)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_RENDERBUFFER, self.rbo3)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_RENDERBUFFER, self.rbo4)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.rbo_dpt)
        gl.glDrawBuffers(5, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2, gl.GL_COLOR_ATTACHMENT3, gl.GL_COLOR_ATTACHMENT4])

        gl.glViewport(0, 0, W, H)  # wtf
        gl.glScissor(0, 0, W, H)  # wtf # NOTE: Need to redefine scissor size

    def resize(self, W=1920, H=1080):
        self.H, self.W = H, W
        if self.H > self.max_H or self.W > self.max_W:
            self.max_H, self.max_W = max(int(self.H * 1.0), self.max_H), max(int(self.W * 1.0), self.max_W)
            self.create_fbo_with_rbos(self.max_W, self.max_H)
        gl.glViewport(0, 0, self.W, self.H)
