import sys

from cuda import cudart

import numpy as np
import cupy as cp

import pyrr
import glfw

from OpenGL.GL import *  # noqa F403
import OpenGL.GL.shaders


def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


class CudaOpenGLMappedBuffer:
    def __init__(self, gl_buffer, flags=0):
        self._gl_buffer = int(gl_buffer)
        self._flags = int(flags)

        self._graphics_ressource = None
        self._cuda_buffer = None

        self.register()

    @property
    def gl_buffer(self):
        return self._gl_buffer

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_ressource(self):
        assert self.registered
        return self._graphics_ressource

    @property
    def registered(self):
        return self._graphics_ressource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def __del__(self):
        self.unregister()

    def register(self):
        if self.registered:
            return self._graphics_ressource
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterBuffer(self._gl_buffer, self._flags)
        )
        return self._graphics_ressource

    def unregister(self):
        if not self.registered:
            return self
        self.unmap()
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsUnregisterResource(self._graphics_ressource)
        )
        return self

    def map(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer

        check_cudart_err(
            cudart.cudaGraphicsMapResources(1, self._graphics_ressource, stream)
        )

        ptr, size = check_cudart_err(
            cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_ressource)
        )

        self._cuda_buffer = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(ptr, size, self), 0
        )

        return self._cuda_buffer

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self

        self._cuda_buffer = check_cudart_err(
            cudart.cudaGraphicsUnmapResources(1, self._graphics_ressource, stream)
        )

        return self


class CudaOpenGLMappedArray(CudaOpenGLMappedBuffer):
    def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order='C'):
        super().__init__(gl_buffer, flags)
        self._dtype = dtype
        self._shape = shape
        self._strides = strides
        self._order = order

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            memptr=self._cuda_buffer,
        )

    def map(self, *args, **kwargs):
        super().map(*args, **kwargs)
        return self.cuda_array


VERTEX_SHADER = """
#version 330
in vec3 position;
uniform mat4 transform;
void main() {
    gl_Position = transform * vec4(position, 1.0f);
}
"""


FRAGMENT_SHADER = """
#version 330
out vec4 outColor;
void main() {
    outColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
"""


def setup_buffers(Nx, Ny):
    max_vertices = Nx * Ny
    max_triangles = 2 * (Nx - 1) * (Ny - 1)

    ftype = np.float32
    itype = np.uint32

    vertex_bytes = 3 * max_vertices * ftype().nbytes
    index_bytes = 3 * max_triangles * itype().nbytes

    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_bytes, None, GL_DYNAMIC_DRAW)
    vertex_buffer = CudaOpenGLMappedArray(ftype, (Ny, Nx, 3), VBO, flags)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_bytes, None, GL_DYNAMIC_DRAW)
    index_buffer = CudaOpenGLMappedArray(itype, (Ny - 1, Nx - 1, 2, 3), EBO, flags)

    Ix = cp.arange(Nx - 1)[None, :]
    Iy = cp.arange(Ny - 1)[:, None]

    I = index_buffer.map()

    I[..., 0, 0] = Iy * Nx + Ix
    I[..., 0, 1] = Iy * Nx + Ix + Nx
    I[..., 0, 2] = Iy * Nx + Ix + Nx + 1
    I[..., 1, 0] = Iy * Nx + Ix + Nx + 1
    I[..., 1, 1] = Iy * Nx + Ix + 1
    I[..., 1, 2] = Iy * Nx + Ix

    index_buffer.unmap()

    return vertex_buffer, index_buffer, max_triangles


def update_vertices(t, vertex_buffer):
    r = (t / 10.0) % 1.0
    r = (1 - r) if r > 0.5 else r

    with vertex_buffer as V:
        (Ny, Nx, _) = V.shape
        θ = cp.linspace(0, np.pi, Nx)[None, :]
        φ = cp.linspace(0, 2 * np.pi, Ny)[:, None]
        V[..., 0] = r * cp.cos(φ) * cp.sin(θ)
        V[..., 1] = r * cp.sin(φ) * cp.sin(θ)
        V[..., 2] = r * cp.cos(θ)


def main():
    if not glfw.init():
        return
    title = "CuPy Cuda/OpenGL interop example"
    window = glfw.create_window(800, 800, title, None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(0)

    shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
    )
    positionLoc = glGetAttribLocation(shader, "position")
    transformLoc = glGetUniformLocation(shader, "transform")

    Nx, Ny = 32, 32
    vertex_buffer, index_buffer, Ntriangles = setup_buffers(Nx, Ny)

    fps = 0
    nframes = 0
    last_time = glfw.get_time()

    glUseProgram(shader)
    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        t = glfw.get_time()
        dt = t - last_time
        if dt >= 1.0:
            fps = nframes / dt
            last_time = t
            nframes = 0

        update_vertices(t, vertex_buffer)

        width, height = glfw.get_window_size(window)
        glViewport(0, 0, width, height)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        rot_x = pyrr.Matrix44.from_x_rotation(0)
        rot_y = pyrr.Matrix44.from_y_rotation(t)

        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer.gl_buffer)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer.gl_buffer)
        glEnableVertexAttribArray(positionLoc)
        glVertexAttribPointer(positionLoc, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glDrawElements(GL_TRIANGLES, int(3 * Ntriangles), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)
        glfw.poll_events()
        glfw.set_window_title(window, f"{title} ({fps:.1f} fps)")
        nframes += 1

    index_buffer.unregister()
    vertex_buffer.unregister()

    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())
