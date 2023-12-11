import torch
import numpy as np
from typing import Tuple, Union
from easyvolcap.utils.console_utils import log, run
from easyvolcap.utils.blend_utils import bilinear_interpolation

# @torch.jit.script # this function is much slower with torch.jit, don't know why yet # FIXME:


def get_ndc_perspective_matrix(K: torch.Tensor,
                               H: int,
                               W: int,
                               n: torch.Tensor = 0.002,  # near far bound
                               f: torch.Tensor = 100,  # near far bound
                               ):
    """
    Note: This is not a academically accurate projection matrix, see z
    Get the perspective matrix that projects camera space points to ndc [-1,1]^3 space
    K[0, 0] and K[1, 1] should be the focal length multiplied by pixel per meter
    x: [-1, -1] should * 2 / W and add W / 2 to be in the center
    y: [-1, -1] should * 2 / H and add H / 2 to be in the center
    z: we're assuming the reciprocal to be with in [-1, 1]

    OpenGL has y going up, x going right and z going inwards window in ndc space
    Our camera is x going right, y going down and z going away from in ndc space
    `nvdiffrast` says its z increases towards the viewer, just like in OpenGL
    And we need to set the actual z to -1/z to get the actual rendering results
    """
    if isinstance(K, torch.Tensor):
        if K.ndim == 3:
            ixt = K.new_zeros(K.shape[0], 4, 4)
        else:
            ixt = K.new_zeros(4, 4)
    elif isinstance(K, np.ndarray):
        if K.ndim == 3:
            ixt = np.zeros((K.shape[0], 4, 4), dtype=K.dtype)
        else:
            ixt = np.zeros((4, 4), dtype=K.dtype)
    else:
        raise NotImplementedError('unsupport data type for K conversion')
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    s = K[..., 0, 1]

    ixt[..., 0, 0] = 2 * fx / W
    ixt[..., 0, 1] = 2 * s / W
    ixt[..., 0, 2] = 1 - 2 * (cx / W)

    ixt[..., 1, 1] = 2 * fy / H
    # ixt[..., 1, 2] = 2 * (cy / H) - 1
    ixt[..., 1, 2] = 1 - 2 * (cy / H)

    ixt[..., 2, 2] = (f + n) / (n - f)
    ixt[..., 2, 3] = 2 * f * n / (n - f)

    ixt[..., 3, 2] = -1

    # ixt[..., 0, 0] = (K[..., 0, 0]) * 2.0 / W  # F * Sx / W * 2
    # ixt[..., 1, 1] = (K[..., 1, 1]) * 2.0 / H  # F * Sy / H * 2
    # ixt[..., 0, 2] = (K[..., 0, 2] - W / 2.0) * 2.0 / W  # Cx / W * 2 - 1
    # ixt[..., 1, 2] = (K[..., 1, 2] - H / 2.0) * 2.0 / H  # Cy / H * 2 - 1
    # ixt[..., 2, 2] = 0
    # ixt[..., 2, 3] = -2 * CLIP_NEAR
    # ixt[..., 3, 2] = 1
    # ixt[..., 3, 3] = 0

    # @property
    # def gl_ixt(self):
    #     # Construct opengl camera matrix with projection & clipping
    #     # https://fruty.io/2019/08/29/augmented-reality-with-opencv-and-opengl-the-tricky-projection-matrix/
    #     # https://gist.github.com/davegreenwood/3a32d779f81f08dce32f3bb423672191
    #     # fmt: off
    #     gl_ixt = mat4(
    #                   2 * self.fx / self.W,                          0,                                       0,  0,
    #                    2 * self.s / self.W,       2 * self.fy / self.H,                                       0,  0,
    #             1 - 2 * (self.cx / self.W), 2 * (self.cy / self.H) - 1,   (self.f + self.n) / (self.n - self.f), -1,
    #                                      0,                          0, 2 * self.f * self.n / (self.n - self.f),  0,
    #     )
    #     # fmt: on

    #     return gl_ixt

    # fx,  0,  0,  cx
    #  0, fy,  0,  cy
    #  0,  0,  0, -2C
    #  0,  0,  1,   0
    return ixt


def make_rotation_y(ry: torch.Tensor):
    Rs = torch.eye(4, 4, device=ry.device)[None].expand(ry.shape[0], -1, -1).clone()  # R, 4, 4 rotation matrix
    siny = ry.sin()
    cosy = ry.cos()
    Rs[:, 0, 0] = cosy
    Rs[:, 0, 2] = siny
    Rs[:, 1, 1] = 1.0
    Rs[:, 2, 0] = -siny
    Rs[:, 2, 2] = cosy

    return Rs  # R, 3, 3


def make_rotation(rx: torch.Tensor = None, ry: torch.Tensor = None, rz: torch.Tensor = None):
    assert not (rx is None and ry is None and rz is None)
    not_none = rx if rx is not None else (ry if ry is not None else rz)
    if rx is None:
        rx = torch.zeros_like(not_none)
    if ry is None:
        ry = torch.zeros_like(not_none)
    if rz is None:
        rz = torch.zeros_like(not_none)

    sinX = rx.sin()
    sinY = ry.sin()
    sinZ = rz.sin()

    cosX = rx.cos()
    cosY = ry.cos()
    cosZ = rz.cos()

    Rx = torch.eye(4, 4, device=ry.device)[None].expand(ry.shape[0], -1, -1).clone()  # R, 4, 4 rotation matrix
    Rx[:, 1, 1] = cosX
    Rx[:, 1, 2] = -sinX
    Rx[:, 2, 1] = sinX
    Rx[:, 2, 2] = cosX

    Ry = torch.eye(4, 4, device=ry.device)[None].expand(ry.shape[0], -1, -1).clone()  # R, 4, 4 rotation matrix
    Ry[:, 0, 0] = cosY
    Ry[:, 0, 2] = sinY
    Ry[:, 2, 0] = -sinY
    Ry[:, 2, 2] = cosY

    Rz = torch.eye(4, 4, device=ry.device)[None].expand(ry.shape[0], -1, -1).clone()  # R, 4, 4 rotation matrix
    Rz[:, 0, 0] = cosZ
    Rz[:, 0, 1] = -sinZ
    Rz[:, 1, 0] = sinZ
    Rz[:, 1, 1] = cosZ

    R = Rz.matmul(Ry).matmul(Rx)
    return R


def render_nvdiffrast(verts: torch.Tensor,
                      faces: torch.IntTensor,
                      attrs: torch.Tensor = None,
                      uv: torch.Tensor = None,
                      img: torch.Tensor = None,
                      #   uvfaces: torch.Tensor = None,
                      H: int = 512,
                      W: int = 512,
                      R: torch.Tensor = None,
                      T: torch.Tensor = None,
                      K: torch.Tensor = None,
                      R_S: int = 2,
                      pos_gradient_boost: float = 1.0,
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rasterize a mesh using nvdiffrast
    Possibly taking uvmap & texture as input, or just vertex color (rgb)

    TODO: implement ranged mode rasterization to get better batching

    verts:   n_batch, n_verts, 3, or n_batch, n_verts, 4 (possible already in clip space or ndc space)
    faces:   n_faces, 3, NOTE: SHARED
    uv:      n_batch, n_uv_verts, 2
    img:     n_batch, tex_h, tex_w, 3, texture
    H:       target image height NOTE: SHARED
    W:       target image width NOTE: SHARED
    R:       n_batch, 3, 3,
    T:       n_batch, 3, 1,
    K:       n_batch, 4, 4,
    R_S:     render scaling, for better gradients

    boost:   apply gradient boost to verts
    """

    if not hasattr(render_nvdiffrast, 'glctx'):
        import nvdiffrast.torch as dr
        try:
            run('python -c \"import nvdiffrast.torch as dr; glctx = dr.RasterizeGLContext(output_db=True)\"', quite=True)  # if this executes without error, we're good to go
            glctx = dr.RasterizeGLContext(output_db=True)  # this will not even raise, the program just crashes
        except RuntimeError as e:
            log('Failed to create OpenGL context, please check your OpenGL installation and Nvidia drivers. Will use CUDA context instead. Note that this might cause performance hits.', 'red')
            glctx = dr.RasterizeCudaContext()
        render_nvdiffrast.glctx = glctx
    glctx = render_nvdiffrast.glctx

    assert uv is not None and img is not None or attrs is not None  # will have to render something out
    assert int(R_S) == R_S  # only supports integer upscaling for now
    # assert not (uvfaces is not None and (uv is None or img is None))

    # support single mesh rendering on multiple views at the same time
    if verts.ndim == 2 and R is not None:
        verts = verts[None].expand(R.shape[0], *verts.shape)
    if attrs.ndim == 2 and R is not None:
        attrs = attrs[None].expand(R.shape[0], *attrs.shape)

    # support changing dtype of faces
    verts = verts.float().contiguous()  # might cause excessive memory usage, after expansion
    attrs = attrs.float().contiguous()  # might cause excessive memory usage, after expansion
    faces = faces.int().contiguous()

    render_attrs = attrs is not None
    render_texture = uv is not None and img is not None
    # unmerge_needed = uvfaces is not None

    B = verts.shape[0]
    if render_attrs:
        A = attrs.shape[-1]

    # if unmerge_needed:
    #     faces, ind_v, ind_uv = unmerge_faces(faces, uvfaces)  # different number of vertices and uv, need to merge them
    #     verts = verts[ind_v]  # 3d locations are also vertex attributes
    #     uv = uv[ind_uv]  # texture coordinate (and its attributes)

    #     if render_attrs:
    #         attrs = attrs[ind_v]  # vertex attributes

    # prepare vertices, apply model view projection transformation maybe
    if R is not None and T is not None:  # prepare intrinsics and extrinsics
        vverts = verts @ R.mT + T.mT
        padding = vverts.new_ones((*vverts.shape[:-1], 1))  # w -> 1
        homovverts = torch.cat([vverts, padding], dim=-1)
    elif verts.shape[-1] == 3:  # manually adding padding of w to homogenccords
        padding = verts.new_ones((*verts.shape[:-1], 1))  # w -> 1
        homovverts = torch.cat([verts, padding], dim=-1)
    else:
        homovverts = verts
        verts = homovverts[..., :-1] / homovverts[..., -1:]

    # projection into NDC space (will be clipped later)
    if K is None:
        K = get_ndc_perspective_matrix(torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -2 * 0.001],
            [0, 0, 1, 0],
        ], device=verts.device, dtype=verts.dtype), H, W)  # 1, 4, 4 -> 1, 4, 4
        ndcverts = homovverts @ K.mT
    else:
        ndcverts = homovverts @ K.mT

    # rasterization (possibly upsampled)
    R_S = int(R_S)
    R_H, R_W = int(H * R_S), int(W * R_S)
    rast, rast_db = dr.rasterize(glctx, ndcverts, faces, resolution=[R_H, R_W])

    # downsampled mask
    rend_msk = rast[:, :, :, 3] != 0
    rast_msk_ind = rend_msk.nonzero(as_tuple=True)
    msk = rend_msk[:, ::R_S, ::R_S].contiguous()  # nearest interpolation
    msk = msk.view(B, H * W)  # flatten

    # perform texture interpolation
    if render_texture and render_attrs:
        interp = torch.cat([attrs, uv], dim=-1)
        interp, puv_db = dr.interpolate(interp, rast, faces, rast_db=rast_db, diff_attrs=[A, A + 1])
        pattrs, puv = interp.split([A, 2], dim=-1)
    elif render_texture and not render_attrs:
        puv, puv_db = dr.interpolate(uv, rast, faces, rast_db=rast_db, diff_attrs=[0, 1])
    else:
        pattrs, pattrs_db = dr.interpolate(attrs, rast, faces, rast_db=rast_db)

    if render_texture:
        prgb = dr.texture(img.flip(1), puv.contiguous(), puv_db.contiguous(), max_mip_level=8)
        # filter unwanted rgb output
        full_prgb = torch.zeros_like(prgb)
        full_prgb[rast_msk_ind] = prgb[rast_msk_ind]
        prgb = full_prgb

    # prepare for anti aliasing
    rend_msk = rend_msk.view(B, R_H, R_W, 1)
    if render_texture and render_attrs:
        aa = torch.cat([pattrs, prgb, rend_msk], dim=-1)
    elif render_texture and not render_attrs:
        aa = torch.cat([prgb, rend_msk], dim=-1)
    else:
        aa = torch.cat([pattrs, rend_msk], dim=-1)

    # perform antialiasing
    aa = dr.antialias(aa, rast, ndcverts, faces, pos_gradient_boost=pos_gradient_boost)  # nvdiffrast wants float tensor as input
    aa = bilinear_interpolation(aa, [H, W])

    # return results
    if render_texture and render_attrs:
        rast_attrs, rast_img, rast_msk = aa.split([A, 3, 1], dim=-1)
    elif render_texture and not render_attrs:
        rast_img, rast_msk = aa.split([3, 1], dim=-1)
    else:
        rast_attrs, rast_msk = aa.split([A, 1], dim=-1)

    rast_msk = rast_msk[..., 0]
    if render_texture and render_attrs:
        return rast_attrs, rast_img, rast_msk
    elif render_texture and not render_attrs:
        return rast_img, rast_msk
    else:
        return rast_attrs, rast_msk
