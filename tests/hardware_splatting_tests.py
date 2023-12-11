# This file tries to render a point cloud with large radius in multiple passes
# And blend them accordingly with the chosen blend function
# This will simulate a manual depth sorting and blending
# I guess hardware are always faster than pure software implementations

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


def test_point_splatting_single_pass():
    render_w = 1024
    render_h = 1024
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    mesh_path = 'assets/meshes/bunny.ply'
    img_path = 'test_point_splatting_single_pass_rgb.png'
    dpt_path = 'test_point_splatting_single_pass_dpt.png'
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
    mesh = Mesh(filename=mesh_path, shade_flat=True, point_radius=0.015)
    mesh.render_type = Mesh.RenderType.POINTS
    mesh.offscreen_render(eglctx, camera)  # TODO: offscreen rendering of points not working, don't know why

    # Read result
    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    img_buf = gl.glReadPixels(0, 0, render_w, render_h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)  # MARK: SYNC
    img = np.frombuffer(img_buf, np.uint8).reshape(render_h, render_w, 4)[::-1]
    save_image(img_path, img)
    log(f'Rendered image saved at: {blue(img_path)}')

    # Read result
    dpt_buf = gl.glReadPixels(0, 0, render_w, render_h, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)  # lost precision... # MARK: SYNC
    dpt = np.frombuffer(dpt_buf, np.float32).reshape(render_h, render_w, 1)[::-1]
    dpt = linearize_depth(dpt, camera.n, camera.f)  # this is the z component in camera coordinates
    dpt = (dpt.min() - dpt) / (dpt.min() - dpt.max())

    save_image(dpt_path, dpt)
    log(f'Rendered depth saved at: {blue(dpt_path)}')


if __name__ == '__main__':
    my_tests(globals())
