from __future__ import absolute_import, division, print_function

# fmt: off
from easyvolcap.utils.egl_utils import eglContextManager  # must be imported before OpenGL.GL
import OpenGL.GL as gl
# fmt: on

import os
import cv2
import torch
import numpy as np
from os.path import join, dirname
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.gl_utils import Quad, Mesh
from easyvolcap.utils.viewer_utils import Camera
from easyvolcap.utils.data_utils import save_image
from easyvolcap.utils.gl_utils import common_opengl_options, linearize_depth
from easyvolcap.utils.test_utils import my_tests

WIDTH, HEIGHT = 512, 512
eglctx = eglContextManager(HEIGHT, WIDTH)  # will create a new context
common_opengl_options()  # common init


def test_gl_context():
    # Render triangle
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glColor3f(1, 0, 0)
    gl.glVertex2f(0, 1)

    gl.glColor3f(0, 1, 0)
    gl.glVertex2f(-1, -1)

    gl.glColor3f(0, 0, 1)
    gl.glVertex2f(1, -1)
    gl.glEnd()

    # Read result
    img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 4)[::-1]

    assert all(img[0, 0, :3] == 0)  # black corner
    assert all(img[0, -1, :3] == 0)  # black corner
    assert img[10, WIDTH // 2, :3].argmax() == 0  # red corner
    assert img[-1, 10, :3].argmax() == 1  # green corner
    assert img[-1, -10, :3].argmax() == 2  # blue corner


def test_gl_mesh_rast():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    mesh_path = 'assets/meshes/bunny.ply'
    img_path = 'test_gl_mesh_rast.png'
    camera = Camera(H=HEIGHT, W=WIDTH,
                    K=torch.tensor([[592., 0., 256.],
                                    [0., 592., 256.],
                                    [0., 0., 1.]]),
                    R=torch.tensor([[0.9908, -0.1353, 0.0000],
                                    [-0.1341, -0.9815, -0.1365],
                                    [0.0185, 0.1353, -0.9906]]),
                    T=torch.tensor([[0.0178],
                                    [0.0953],
                                    [0.3137]])
                    )
    mesh = Mesh(filename=mesh_path, shade_flat=False)
    mesh.render(camera)
    # ndc = torch.cat([mesh.verts, torch.ones_like(mesh.verts)[..., -1:]], dim=-1).numpy() @ glm.transpose(camera.gl_ext) @ glm.transpose(camera.gl_ixt)

    # Read result
    img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 4)[::-1]
    save_image(img_path, img)
    log(f'Rendered image saved at: {blue(img_path)}')


def test_gl_tex_blit():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    quad = Quad(H=HEIGHT, W=WIDTH)
    data = np.random.randint(0, 255, (HEIGHT, WIDTH, 4), dtype=np.uint8)
    data[20:30, 20:30] = [255, 0, 0, 255]
    data[30:40, 20:30] = [0, 255, 0, 255]
    data[20:30, 30:40] = [0, 0, 255, 255]
    data[30:40, 30:40] = [255, 255, 255, 255]
    data = data[::-1]  # flip 0
    data = np.asarray(data, order='C')

    quad.upload_to_texture(data)  # upload the pointer

    # FIXME: TEST FAILING AFTER CHANING ALIGNMENT
    fbo = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo)
    gl.glFramebufferTexture2D(gl.GL_READ_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, quad.tex, 0)  # attach the texture to the framebuffer as logical buffer
    gl.glBlitFramebuffer(0, 0, WIDTH, HEIGHT,
                         0, 0, WIDTH, HEIGHT,
                         gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, eglctx.fbo)  # TODO: DEBUG LAG

    # Read result
    img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)  # TODO: FIX THIS
    img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 4)[::-1]
    img_path = 'test_gl_tex_blit.png'
    save_image(img_path, img)
    log(f'Tex blit image saved at: {blue(img_path)}')

    assert img[20, 20, :3].argmax() == 0  # red corner
    assert img[30, 20, :3].argmax() == 1  # green corner
    assert img[20, 30, :3].argmax() == 2  # blue corner
    assert all(img[30, 30] == 255)  # white corner


def test_gl_quad_blit():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    quad = Quad(H=HEIGHT, W=WIDTH)
    data = np.random.randint(0, 255, (HEIGHT, WIDTH, 4), dtype=np.uint8)

    data[10:20, 10:20] = [255, 0, 0, 255]
    data[20:30, 10:20] = [0, 255, 0, 255]
    data[10:20, 20:30] = [0, 0, 255, 255]
    data[20:30, 20:30] = [255, 255, 255, 255]
    data = data[::-1]  # flip 0
    data = np.asarray(data, order='C')

    quad.upload_to_texture(data)  # upload the pointer
    quad.blit()  # no camera to render

    # Read result
    img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 4)[::-1]

    img_path = 'test_gl_quad_blit.png'
    save_image(img_path, img)
    log(f'Quad blit image saved at: {blue(img_path)}')

    assert img[10, 10, :3].argmax() == 0  # red corner
    assert img[20, 10, :3].argmax() == 1  # green corner
    assert img[10, 20, :3].argmax() == 2  # blue corner
    assert all(img[20, 20] == 255)  # white corner


def test_gl_quad_draw():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    quad = Quad(H=HEIGHT, W=WIDTH)
    data = np.random.randint(0, 255, (HEIGHT, WIDTH, 4), dtype=np.uint8)

    data[-20:-10, -20:-10] = [255, 0, 0, 255]
    data[-30:-20, -20:-10] = [0, 255, 0, 255]
    data[-20:-10, -30:-20] = [0, 0, 255, 255]
    data[-30:-20, -30:-20] = [255, 255, 255, 255]
    data = data[::-1]  # flip 0
    data = np.asarray(data, order='C')

    quad.upload_to_texture(data)  # upload the pointer
    quad.draw()  # no camera to render

    # Read result
    img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 4)[::-1]

    img_path = 'test_gl_quad_draw.png'
    save_image(img_path, img)
    log(f'Quad draw image saved at: {blue(img_path)}')

    assert img[-20, -20, :3].argmax() == 0  # red corner
    assert img[-30, -20, :3].argmax() == 1  # green corner
    assert img[-20, -30, :3].argmax() == 2  # blue corner
    assert all(img[-30, -30] == 255)  # white corner


def test_eglctx_manual_fbo_rbo():  # TODO: BLACK RESULTS ON WSL (BUT COULD RENDER WITHOUT RESIZING)
    render_w = 1024
    render_h = 1024
    # eglctx.resize(render_w, render_h)
    gl.glViewport(0, 0, render_w, render_h)  # wtf
    gl.glScissor(0, 0, render_w, render_h)  # wtf # NOTE: Need to redefine scissor size
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Add new buffer
    old = gl.glGetInteger(gl.GL_FRAMEBUFFER_BINDING)
    fbo = gl.glGenFramebuffers(1)
    rbo0, rbo1, rbo_dpt = gl.glGenRenderbuffers(3)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rbo0)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, render_w, render_h)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rbo1)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, render_w, render_h)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rbo_dpt)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, render_w, render_h)

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, rbo0)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_RENDERBUFFER, rbo1)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, rbo_dpt)
    gl.glDrawBuffers(2, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1])

    mesh_path = 'assets/meshes/bunny.ply'
    img_path = 'test_eglctx_manual_fbo_rbo.png'
    dpt_path = 'test_eglctx_manual_fbo_rbo_dpt.png'
    camera = Camera(H=render_h, W=render_w,
                    K=torch.tensor([[render_w * 592 / 512, 0., render_w / 2],
                                    [0., render_h * 592 / 512, render_h / 2],
                                    [0., 0., 1.]]),
                    R=torch.tensor([[0.9908, -0.1353, 0.0000],
                                    [-0.1341, -0.9815, -0.1365],
                                    [0.0185, 0.1353, -0.9906]]),
                    T=torch.tensor([[0.0178],
                                    [0.0953],
                                    [0.3137]]),
                    n=0.02,
                    f=1,
                    )
    mesh = Mesh(filename=mesh_path, shade_flat=False)
    # mesh.offscreen_render(eglctx, camera)
    mesh.render(camera)

    # Read result
    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    img_buf = gl.glReadPixels(0, 0, render_w, render_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(img_buf, np.uint8).reshape(render_h, render_w, 4)[::-1]
    save_image(img_path, img)
    log(f'Rendered image saved at: {blue(img_path)}')

    # Read result
    # gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
    # dpt_buf = gl.glReadPixels(0, 0, render_w, render_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)  # lost precision...
    # dpt = np.frombuffer(dpt_buf, np.uint8).reshape(render_h, render_w, 4)[::-1]
    # dpt = dpt[..., 0] / 1 + dpt[..., 0] / 256 + dpt[..., 0] / 65536 + dpt[..., 0] / 16777216
    # dpt = (dpt - dpt.min()) / (dpt.max() - dpt.min())
    dpt_buf = gl.glReadPixels(0, 0, render_w, render_h, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)  # lost precision... # MARK: SYNC
    # gl.glBindTexture(gl.GL_TEXTURE_2D, eglctx.tex_dpt)
    # dpt_buf = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
    dpt = np.frombuffer(dpt_buf, np.float32).reshape(render_h, render_w, 1)[::-1]
    dpt = linearize_depth(dpt, camera.n, camera.f)
    dpt = (dpt.min() - dpt) / (dpt.min() - dpt.max())
    save_image(dpt_path, dpt)
    log(f'Rendered depth saved at: {blue(dpt_path)}')

    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, old)


def test_eglctx_auto_fbo_rbo():
    render_w = 1024
    render_h = 1024
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    mesh_path = 'assets/meshes/bunny.ply'
    img_path = 'test_eglctx_auto_fbo_rbo.png'
    dpt_path = 'test_eglctx_auto_fbo_rbo_dpt.png'
    camera = Camera(H=render_h, W=render_w,
                    K=torch.tensor([[render_w * 592 / 512, 0., render_w / 2],
                                    [0., render_h * 592 / 512, render_h / 2],
                                    [0., 0., 1.]]),
                    R=torch.tensor([[0.9908, -0.1353, 0.0000],
                                    [-0.1341, -0.9815, -0.1365],
                                    [0.0185, 0.1353, -0.9906]]),
                    T=torch.tensor([[0.0178],
                                    [0.0953],
                                    [0.3137]]),
                    n=0.02,
                    f=1,
                    )
    mesh = Mesh(filename=mesh_path, shade_flat=False)
    mesh.offscreen_render(eglctx, camera)

    # Read result
    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    img_buf = gl.glReadPixels(0, 0, render_w, render_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(img_buf, np.uint8).reshape(render_h, render_w, 4)[::-1]
    save_image(img_path, img)
    log(f'Rendered image saved at: {blue(img_path)}')

    # Read result
    # gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
    # dpt_buf = gl.glReadPixels(0, 0, render_w, render_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)  # lost precision...
    # dpt = np.frombuffer(dpt_buf, np.uint8).reshape(render_h, render_w, 4)[::-1]
    # dpt = dpt[..., 0] / 1 + dpt[..., 0] / 256 + dpt[..., 0] / 65536 + dpt[..., 0] / 16777216
    dpt_buf = gl.glReadPixels(0, 0, render_w, render_h, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)  # lost precision... # MARK: SYNC
    # gl.glBindTexture(gl.GL_TEXTURE_2D, eglctx.tex_dpt)
    # dpt_buf = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
    dpt = np.frombuffer(dpt_buf, np.float32).reshape(render_h, render_w, 1)[::-1]
    dpt = linearize_depth(dpt, camera.n, camera.f)  # this is the z component in camera coordinates
    dpt = (dpt.min() - dpt) / (dpt.min() - dpt.max())

    save_image(dpt_path, dpt)
    log(f'Rendered depth saved at: {blue(dpt_path)}')


if __name__ == '__main__':
    my_tests(globals())
