import os
import re
import cv2
import h5py
import torch
import struct
import asyncio
import subprocess
import numpy as np

from PIL import Image
from io import BytesIO
from typing import overload
from functools import lru_cache

# from imgaug import augmenters as iaa
from typing import Tuple, Union, List, Dict

from torch.nn import functional as F
from torch.utils.data._utils.pin_memory import pin_memory
from torch.utils.data._utils.collate import default_collate, default_convert

from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *

from enum import Enum, auto

# Copied from enerf (maybe was in turn copied from dtu)


def read_pickle(name):
    import pickle
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    return intrinsics, extrinsics, depth_min


def read_pmn_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[1])
    return intrinsics, extrinsics, depth_min, depth_max


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def generate_video(result_str: str,
                   output: str,
                   fps: int = 30,
                   crf: int = 17,
                   cqv: int = 19,
                   lookahead: int = 20,
                   hwaccel: str = 'cuda',
                   preset: str = 'p7',
                   tag: str = 'hvc1',
                   vcodec: str = 'hevc_nvenc',
                   pix_fmt: str = 'yuv420p',  # chrome friendly
                   ):
    cmd = [
        'ffmpeg',
        '-hwaccel', hwaccel,
        '-hide_banner',
        '-loglevel', 'error',
        '-framerate', fps,
        '-f', 'image2',
        '-pattern_type', 'glob',
        '-nostdin',  # otherwise you cannot chain commands together
        '-y',
        '-r', fps,
        '-i', result_str,
        '-c:v', vcodec,
        '-preset', preset,
        '-cq:v', cqv,
        '-rc:v', 'vbr',
        '-tag:v', tag,
        '-crf', crf,
        '-pix_fmt', pix_fmt,
        '-rc-lookahead', lookahead,
        '-vf', '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',  # avoid yuv420p odd number bug
        output,
    ]
    run(cmd)
    return output


def numpy_to_video(numpy_array: np.ndarray,
                   output_filename: str,
                   fps: float = 30.0,
                   crf: int = 18,
                   cqv: int = 19,
                   lookahead: int = 20,
                   preset='veryslow',
                   vcodec='libx265',
                   ):
    """
    Convert a numpy array (T, H, W, C) to a video using ffmpeg.

    Parameters:
    - numpy_array: Numpy array to be converted.
    - output_filename: The filename of the output video.
    - framerate: Frame rate for the video.
    """
    if isinstance(numpy_array, np.ndarray):
        T, H, W, C = numpy_array.shape
    else:
        T = len(numpy_array)
        H, W, C = numpy_array[0].shape
    assert C == 3, "Expected 3 channels!"

    cmd = [
        'ffmpeg',
        '-hwaccel', 'cuda',
        '-v', 'quiet', '-stats',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',  # Size of one frame
        '-pix_fmt', 'rgb24',
        '-r', fps,  # Frame rate
        '-i', '-',  # Read from pipe
        '-an',  # No audio
        '-vcodec', vcodec,
        '-preset', preset,
        '-cq:v', cqv,
        '-crf', crf,
        '-rc-lookahead', lookahead,
        '-rc:v', 'vbr',
        '-tag:v', 'hvc1',
        output_filename
    ]
    os.makedirs(dirname(output_filename), exist_ok=True)
    process = subprocess.Popen(map(str, cmd), stdin=subprocess.PIPE)
    # process.communicate(input=numpy_array.tobytes())
    for frame in numpy_array:
        process.stdin.write(frame.tobytes())
        # process.stdin.flush()
    process.stdin.close()
    process.communicate()


def get_video_dimensions(input_filename):
    """
    Extract the width and height of a video using ffprobe.

    Parameters:
    - input_filename: The filename of the input video.

    Returns:
    - width and height of the video.
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        input_filename
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = pipe.communicate()
    width, height = map(int, out.decode('utf-8').strip().split('x'))
    return width, height


def video_to_numpy(input_filename, hwaccel='cuda', vcodec='hevc_cuvid'):
    """
    Convert a video file to a numpy array (T, H, W, C) using ffmpeg.

    Parameters:
    - input_filename: The filename of the input video.

    Returns:
    - Numpy array representing the video.
    """
    W, H = get_video_dimensions(input_filename)

    cmd = [
        'ffmpeg',
    ]
    if hwaccel != 'none':
        cmd += ['-hwaccel', hwaccel,]
    cmd += [
        '-v', 'quiet', '-stats',
    ]
    if vcodec != 'none':
        cmd += ['-vcodec', vcodec,]
    cmd += [
        '-i', input_filename,
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ]

    pipe = subprocess.Popen(map(str, cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    raw_data, _ = pipe.communicate()

    # Convert the raw data to numpy array and reshape
    video_np = np.frombuffer(raw_data, dtype=np.uint8)
    H2, W2 = (H + 1) // 2 * 2, (W + 1) // 2 * 2
    try:
        video_np = video_np.reshape(-1, H2, W2, 3)[:, :H, :W, :]
    except ValueError as e:
        video_np = video_np.reshape(-1, H, W, 3)

    return video_np


class Visualization(Enum):
    # Universal visualization
    RENDER = auto()  # plain rgb render output
    SURFACE = auto()  # surface position (similar to depth)
    DEFORM = auto()  # deformation magnitude (as in correspondence?)
    DEPTH = auto()  # needs a little bit extra computation
    ALPHA = auto()  # occupancy (rendered volume density)
    NORMAL = auto()  # needs extra computation
    FEATURE = auto()  # embedder results
    SEMANTIC = auto()  # semantic nerf related
    SRCINPS = auto()  # Souce input images for image based rendering

    # jacobian related
    JACOBIAN = auto()

    # Relighting related
    ENVMAP = auto()
    ALBEDO = auto()
    SHADING = auto()
    ROUGHNESS = auto()

    # Geometry related output
    MESH = auto()
    POINT = auto()
    VOLUME = auto()


class DataSplit(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


def variance_of_laplacian(image: np.ndarray):
    if image.ndim == 3 and image.shape[-1] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def alpha2sdf(alpha, beta, dists=0.005):
    return beta * np.log(2 * beta * (-np.log(1 - alpha) / dists))


def h5_to_dotdict(h5: h5py.File) -> dotdict:
    d = {key: h5_to_dotdict(h5[key]) if isinstance(h5[key], h5py.Group) else h5[key][:] for key in h5.keys()}  # loaded as numpy array
    d = dotdict(d)
    return d


def h5_to_list_of_dotdict(h5: h5py.File) -> list:
    return [h5_to_dotdict(h5[key]) for key in tqdm(h5)]


def to_h5py(value, h5: h5py.File, key: str = None, compression: str = 'gzip'):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        h5.create_dataset(str(key), data=value, compression=compression)
    elif isinstance(value, list):
        if key is not None:
            h5 = h5.create_group(str(key))
        [to_h5py(v, h5, k) for k, v in enumerate(value)]
    elif isinstance(value, dict):
        if key is not None:
            h5 = h5.create_group(str(key))
        [to_h5py(v, h5, k) for k, v in value.items()]
    else:
        raise NotImplementedError(f'unsupported type to write to h5: {type(value)}')


def export_h5(batch: dotdict, filename):
    with h5py.File(filename, 'w') as f:
        to_h5py(batch, f)


def load_h5(filename):
    with h5py.File(filename, 'r') as f:
        return h5_to_dotdict(f)


def merge_faces(faces, *args):
    # Copied from trimesh, this will select one uv coordinates for a particular vertex
    """
    Textured meshes can come with faces referencing vertex
    indices (`v`) and an array the same shape which references
    vertex texture indices (`vt`) and sometimes even normal (`vn`).

    Vertex locations with different values of any of these can't
    be considered the "same" vertex, and for our simple data
    model we need to not combine these vertices.

    Parameters
    -------------
    faces : (n, d) int
      References vertex indices
    *args : (n, d) int
      Various references of corresponding values
      This is usually UV coordinates or normal indexes
    maintain_faces : bool
      Do not alter original faces and return no-op masks.

    Returns
    -------------
    new_faces : (m, d) int
      New faces for masked vertices
    mask_v : (p,) int
      A mask to apply to vertices
    mask_* : (p,) int
      A mask to apply to vt array to get matching UV coordinates
      Returns as many of these as args were passed
    """

    # start with not altering faces at all
    result = [faces]
    # find the maximum index referenced by faces
    max_idx = faces.max()
    # add a vertex mask which is just ordered
    result.append(np.arange(max_idx + 1))

    # now given the order is fixed do our best on the rest of the order
    for arg in args:
        # create a mask of the attribute-vertex mapping
        # note that these might conflict since we're not unmerging
        masks = np.zeros((3, max_idx + 1), dtype=np.int64)
        # set the mask using the unmodified face indexes
        for i, f, a in zip(range(3), faces.T, arg.T):
            masks[i][f] = a
        # find the most commonly occurring attribute (i.e. UV coordinate)
        # and use that index note that this is doing a float conversion
        # and then median before converting back to int: could also do this as
        # a column diff and sort but this seemed easier and is fast enough
        result.append(np.median(masks, axis=0).astype(np.int64))

    return result


def get_mesh(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, colors: torch.Tensor = None, normals: torch.Tensor = None, filename: str = "default.ply"):
    from trimesh import Trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial, SimpleMaterial
    from easyvolcap.utils.mesh_utils import face_normals, loop_subdivision

    verts, faces = to_numpy([verts, faces])
    verts = verts.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    # MARK: used process=False here to preserve vertex order
    mesh = Trimesh(verts, faces, process=False)
    if colors is None:
        # colors = verts
        colors = face_normals(torch.from_numpy(verts), torch.from_numpy(faces).long()) * 0.5 + 0.5
    colors = to_numpy(colors)
    colors = colors.reshape(-1, 3)
    colors = (np.concatenate([colors, np.ones([*colors.shape[:-1], 1])], axis=-1) * 255).astype(np.uint8)
    if len(verts) == len(colors):
        mesh.visual.vertex_colors = colors
    elif len(faces) == len(colors):
        mesh.visual.face_colors = colors

    if normals is not None:
        normals = to_numpy(normals)
        mesh.vertex_normals = normals

    if uv is not None:
        uv = to_numpy(uv)
        uv = uv.reshape(-1, 2)
        img = to_numpy(img)
        img = img.reshape(*img.shape[-3:])
        img = Image.fromarray(np.uint8(img * 255))
        mat = SimpleMaterial(
            image=img,
            diffuse=(0.8, 0.8, 0.8),
            ambient=(1.0, 1.0, 1.0),
        )
        mat.name = os.path.splitext(os.path.split(filename)[1])[0]
        texture = TextureVisuals(uv=uv, material=mat)
        mesh.visual = texture

    return mesh


def get_tensor_mesh_data(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, uvfaces: torch.Tensor = None):

    # pytorch3d wants a tensor
    verts, faces, uv, img, uvfaces = to_tensor([verts, faces, uv, img, uvfaces])
    verts = verts.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    uv = uv.reshape(-1, 2)
    img = img.reshape(img.shape[-3:])
    uvfaces = uvfaces.reshape(-1, 3)

    # textures = TexturesUV(img, uvfaces, uv)
    # meshes = Meshes(verts, faces, textures)
    return verts, faces, uv, img, uvfaces


def export_npz(batch: dotdict, filename: struct):
    export_dotdict(batch, filename)


def export_dotdict(batch: dotdict, filename: struct):
    batch = to_numpy(batch)
    np.savez_compressed(filename, **batch)


def load_mesh(filename: str, device='cuda', load_uv=False, load_aux=False, backend='pytorch3d'):
    from pytorch3d.io import load_ply, load_obj
    if backend == 'trimesh':
        import trimesh
        mesh: trimesh.Trimesh = trimesh.load(filename)
        return mesh.vertices, mesh.faces

    vm, fm = None, None
    if filename.endswith('.npz'):
        mesh = np.load(filename)
        v = torch.from_numpy(mesh['verts'])
        f = torch.from_numpy(mesh['faces'])

        if load_uv:
            vm = torch.from_numpy(mesh['uvs'])
            fm = torch.from_numpy(mesh['uvfaces'])
    else:
        if filename.endswith('.ply'):
            v, f = load_ply(filename)
        elif filename.endswith('.obj'):
            v, faces_attr, aux = load_obj(filename)
            f = faces_attr.verts_idx

            if load_uv:
                vm = aux.verts_uvs
                fm = faces_attr.textures_idx
        else:
            raise NotImplementedError(f'Unrecognized input format for: {filename}')

    v = v.to(device, non_blocking=True).contiguous()
    f = f.to(device, non_blocking=True).contiguous()

    if load_uv:
        vm = vm.to(device, non_blocking=True).contiguous()
        fm = fm.to(device, non_blocking=True).contiguous()

    if load_uv:
        if load_aux:
            return v, f, vm, fm, aux
        else:
            return v, f, vm, fm
    else:
        return v, f


def load_pts(filename: str):
    from pyntcloud import PyntCloud
    cloud = PyntCloud.from_file(filename)
    verts = cloud.xyz
    if 'red' in cloud.points and 'green' in cloud.points and 'blue' in cloud.points:
        r = np.asarray(cloud.points['red'])
        g = np.asarray(cloud.points['green'])
        b = np.asarray(cloud.points['blue'])
        colors = (np.stack([r, g, b], axis=-1) / 255).astype(np.float32)
    elif 'r' in cloud.points and 'g' in cloud.points and 'b' in cloud.points:
        r = np.asarray(cloud.points['r'])
        g = np.asarray(cloud.points['g'])
        b = np.asarray(cloud.points['b'])
        colors = (np.stack([r, g, b], axis=-1) / 255).astype(np.float32)
    else:
        colors = None

    if 'nx' in cloud.points and 'ny' in cloud.points and 'nz' in cloud.points:
        nx = np.asarray(cloud.points['nx'])
        ny = np.asarray(cloud.points['ny'])
        nz = np.asarray(cloud.points['nz'])
        norms = np.stack([nx, ny, nz], axis=-1)
    else:
        norms = None

    # if 'alpha' in cloud.points:
    #     cloud.points['alpha'] = cloud.points['alpha'] / 255

    reserved = ['x', 'y', 'z', 'red', 'green', 'blue', 'r', 'g', 'b', 'nx', 'ny', 'nz']
    scalars = dotdict({k: np.asarray(cloud.points[k])[..., None] for k in cloud.points if k not in reserved})  # one extra dimension at the back added
    return verts, colors, norms, scalars


def export_pts(pts: torch.Tensor, color: torch.Tensor = None, normal: torch.Tensor = None, scalars: dotdict = dotdict(), filename: str = "default.ply"):
    from pandas import DataFrame
    from pyntcloud import PyntCloud

    data = dotdict()
    pts = to_numpy(pts)  # always blocking?
    pts = pts.reshape(-1, 3)
    data.x = pts[:, 0].astype(np.float32)
    data.y = pts[:, 1].astype(np.float32)
    data.z = pts[:, 2].astype(np.float32)

    if color is not None:
        color = to_numpy(color)
        color = color.reshape(-1, 3)
        data.red = (color[:, 0] * 255).astype(np.uint8)
        data.green = (color[:, 1] * 255).astype(np.uint8)
        data.blue = (color[:, 2] * 255).astype(np.uint8)
    else:
        data.red = (pts[:, 0] * 255).astype(np.uint8)
        data.green = (pts[:, 1] * 255).astype(np.uint8)
        data.blue = (pts[:, 2] * 255).astype(np.uint8)

    # if 'alpha' in scalars:
    #     data.alpha = (scalars.alpha * 255).astype(np.uint8)

    if normal is not None:
        normal = to_numpy(normal)
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-13)
        normal = normal.reshape(-1, 3)
        data.nx = normal[:, 0].astype(np.float32)
        data.ny = normal[:, 1].astype(np.float32)
        data.nz = normal[:, 2].astype(np.float32)

    if scalars is not None:
        scalars = to_numpy(scalars)
        for k, v in scalars.items():
            v = v.reshape(-1, 1)
            data[k] = v[:, 0]

    df = DataFrame(data)
    cloud = PyntCloud(df)  # construct the data
    dir = dirname(filename)
    if dir: os.makedirs(dir, exist_ok=True)
    return cloud.to_file(filename)


def export_lines(verts: torch.Tensor, lines: torch.Tensor, color: torch.Tensor = None, filename: str = 'default.ply'):
    if color is None:
        color = verts
    verts, lines, color = to_numpy([verts, lines, color])  # always blocking?
    if color.dtype == np.float32:
        color = (color * 255).astype(np.uint8)
    verts = verts.reshape(-1, 3)
    lines = lines.reshape(-1, 2)
    color = color.reshape(-1, 3)

    # Write to PLY
    with open(filename, 'wb') as f:
        # PLY header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(verts)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(f"element edge {len(lines)}\n".encode())
        f.write(b"property int vertex1\n")
        f.write(b"property int vertex2\n")
        f.write(b"end_header\n")

        # Write vertices and colors
        for v, c in zip(verts, color):
            f.write(struct.pack('fffBBB', v[0], v[1], v[2], c[0], c[1], c[2]))

        # Write lines
        for l in lines:
            f.write(struct.pack('ii', l[0], l[1]))


def export_camera(c2w: torch.Tensor, ixt: torch.Tensor = None, col: torch.Tensor = torch.tensor([50, 50, 200]), axis_size=0.10, filename: str = 'default.ply'):
    verts = []
    lines = []
    rgbs = []

    def add_line(p0: torch.Tensor, p1: torch.Tensor, col: torch.Tensor):
        # Add a and b vertices
        verts.append(p0)  # N, M, 3
        verts.append(p1)  # N, M, 3
        sh = p0.shape[:-1]

        # Add the vertex colors
        col = torch.broadcast_to(col, sh + (3,))
        rgbs.append(col)
        rgbs.append(col)

        # Add the faces
        new = p0.numel() // 3  # number of new elements
        curr = new * (len(verts) - 2)  # assume all previous elements are of the same size
        start = torch.arange(curr, curr + new)
        end = torch.arange(curr + new, curr + new * 2)
        line = torch.stack([start, end], dim=-1)  # NM, 2
        line = line.view(sh + (2,))
        lines.append(line)

    c2w = c2w[..., :3, :]
    p = c2w[..., 3]  # third row (corresponding to 3rd column)

    if ixt is None: aspect = 1.0
    else: aspect = ixt[..., 0, 0][..., None] / ixt[..., 1, 1][..., None]
    if ixt is None: focal = 1000
    else: focal = (ixt[..., 0, 0][..., None] + ixt[..., 1, 1][..., None]) / 2

    axis_size = focal * axis_size / 1000
    xs = axis_size * aspect
    ys = axis_size
    zs = axis_size * aspect * 2

    a = p + xs * c2w[..., 0] + ys * c2w[..., 1] + zs * c2w[..., 2]
    b = p - xs * c2w[..., 0] + ys * c2w[..., 1] + zs * c2w[..., 2]
    c = p - xs * c2w[..., 0] - ys * c2w[..., 1] + zs * c2w[..., 2]
    d = p + xs * c2w[..., 0] - ys * c2w[..., 1] + zs * c2w[..., 2]

    add_line(p, p + axis_size * c2w[..., 0], torch.tensor([255, 64, 64]))
    add_line(p, p + axis_size * c2w[..., 1], torch.tensor([64, 255, 64]))
    add_line(p, p + axis_size * c2w[..., 2], torch.tensor([64, 64, 255]))
    add_line(p, a, col)
    add_line(p, b, col)
    add_line(p, c, col)
    add_line(p, d, col)
    add_line(a, b, col)
    add_line(b, c, col)
    add_line(c, d, col)
    add_line(d, a, col)

    verts = torch.stack(verts)
    lines = torch.stack(lines)
    rgbs = torch.stack(rgbs)

    export_lines(verts, lines, rgbs, filename=filename)


def export_mesh(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, uvfaces: torch.Tensor = None, colors: torch.Tensor = None, normals: torch.Tensor = None, filename: str = "default.ply", subdivision=0):
    if dirname(filename): os.makedirs(dirname(filename), exist_ok=True)

    if subdivision > 0:
        from easyvolcap.utils.mesh_utils import face_normals, loop_subdivision
        verts, faces = loop_subdivision(verts, faces, subdivision)

    if filename.endswith('.npz'):
        def collect_args(**kwargs): return kwargs
        kwargs = collect_args(verts=verts, faces=faces, uv=uv, img=img, uvfaces=uvfaces, colors=colors, normals=normals)
        ret = dotdict({k: v for k, v in kwargs.items() if v is not None})
        export_dotdict(ret, filename)

    elif filename.endswith('.ply') or filename.endswith('.obj'):
        if uvfaces is None:
            mesh = get_mesh(verts, faces, uv, img, colors, normals, filename)
            mesh.export(filename)
        else:
            from pytorch3d.io import save_obj
            verts, faces, uv, img, uvfaces = get_tensor_mesh_data(verts, faces, uv, img, uvfaces)
            save_obj(filename, verts, faces, verts_uvs=uv, faces_uvs=uvfaces, texture_map=img)
    else:
        raise NotImplementedError(f'Unrecognized input format for: {filename}')


def export_pynt_pts_alone(pts, color=None, filename="default.ply"):
    import pandas as pd
    from pyntcloud import PyntCloud
    data = {}

    pts = pts if isinstance(pts, np.ndarray) else pts.detach().cpu().numpy()
    pts = pts.reshape(-1, 3)
    data['x'] = pts[:, 0].astype(np.float32)
    data['y'] = pts[:, 1].astype(np.float32)
    data['z'] = pts[:, 2].astype(np.float32)

    if color is not None:
        color = color if isinstance(color, np.ndarray) else color.detach().cpu().numpy()
        color = color.reshape(-1, 3)
        data['red'] = color[:, 0].astype(np.uint8)
        data['green'] = color[:, 1].astype(np.uint8)
        data['blue'] = color[:, 2].astype(np.uint8)
    else:
        data['red'] = (pts[:, 0] * 255).astype(np.uint8)
        data['green'] = (pts[:, 1] * 255).astype(np.uint8)
        data['blue'] = (pts[:, 2] * 255).astype(np.uint8)

    df = pd.DataFrame(data)
    cloud = PyntCloud(df)  # construct the data
    dirname = dirname(filename)
    if dirname: os.makedirs(dirname, exist_ok=True)
    return cloud.to_file(filename)


def export_o3d_pts(pts: torch.Tensor, filename: str = "default.ply"):
    import open3d as o3d
    pts = to_numpy(pts)
    pts = pts.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return o3d.io.write_point_cloud(filename, pcd)


def export_o3d_pcd(pts: torch.Tensor, rgb: torch.Tensor, normal: torch.Tensor, filename="default.ply"):
    import open3d as o3d
    pts, rgb, normal = to_numpy([pts, rgb, normal])
    pts = pts.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    normal = normal.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    return o3d.io.write_point_cloud(filename, pcd)


def export_pcd(pts: torch.Tensor, rgb: torch.Tensor, occ: torch.Tensor, filename="default.ply"):
    import pandas as pd
    from pyntcloud import PyntCloud
    pts, rgb, occ = to_numpy([pts, rgb, occ])
    pts = pts.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    occ = occ.reshape(-1, 1)
    # MARK: CloudCompare bad, set first to 0, last to 1
    for i in range(3):
        rgb[0, i] = 0
        rgb[-1, i] = 1
    occ[0, 0] = 0
    occ[-1, 0] = 1

    data = dotdict()
    data.x = pts[:, 0]
    data.y = pts[:, 1]
    data.z = pts[:, 2]
    # TODO: maybe, for compability, save color as uint?
    # currently saving as float number from [0, 1]
    data.red = rgb[:, 0]
    data.green = rgb[:, 1]
    data.blue = rgb[:, 2]
    data.alpha = occ[:, 0]

    # MARK: We're saving extra scalars for loading in CloudCompare
    # can't assign same property to multiple fields
    data.r = rgb[:, 0]
    data.g = rgb[:, 1]
    data.b = rgb[:, 2]
    data.a = occ[:, 0]
    df = pd.DataFrame(data)

    cloud = PyntCloud(df)  # construct the data
    dirname = dirname(filename)
    if dirname: os.makedirs(dirname, exist_ok=True)
    return cloud.to_file(filename)


def load_rgb_image(img_path) -> np.ndarray:
    # return cv2.imread(img_path, cv2.IMREAD_COLOR)[..., ::-1].copy()  # removing the stride (for conversion to tensor)
    return cv2.imread(img_path, cv2.IMREAD_COLOR)[..., [2, 1, 0]]  # BGR to RGB


def load_unchanged_image(img_path) -> np.ndarray:
    return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


def load_npz(index, folder):
    path = os.path.join(folder, f"{index}.npz")
    data = np.load(path)
    return dotdict({**data})


def load_dotdict(path):
    f = np.load(path)
    f = dotdict({**f})
    return f


def start_save_npz(index, dir, param: dict, remove_batch=True):
    return asyncio.create_task(async_save_npz(index, dir, param, remove_batch))


async def async_save_npz(index, dir, param: dict, remove_batch=True):
    log(f"Trying to save: {index}")
    save_npz(index, dir, param, remove_batch)


def save_img(index, dir, img: torch.Tensor, remove_batch=True, remap=False, flip=False):

    img = to_numpy(img)

    if remap:
        img *= 255
        img = img.astype(np.uint8)
    if flip:
        img = img[..., ::-1]

    if remove_batch:
        n_batch = img.shape[0]
        for b in range(n_batch):
            file_path = os.path.join(dir, f"{index*n_batch + b}.png")
            im = img[b]
            cv2.imwrite(file_path, im)
    else:
        file_path = os.path.join(dir, f"{index}.png")
        cv2.imwrite(file_path, img)


def save_npz(index, dir, param: dict, remove_batch=False):
    param = to_numpy(param)
    if remove_batch:
        n_batch = param[next(iter(param))].shape[0]
        for b in range(n_batch):
            file_path = os.path.join(dir, f"{index*n_batch + b}.npz")
            p = {k: v[b] for k, v in param.items()}
            np.savez_compressed(file_path, **p)
    else:
        file_path = os.path.join(dir, f"{index}.npz")
        np.savez_compressed(file_path, **param)


def to_cuda(batch, device="cuda", ignore_list: bool = False) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = [to_cuda(b, device, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_cuda(v, device, ignore_list) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device, non_blocking=True)
    else:  # numpy and others
        batch = torch.as_tensor(batch, device=device)
    return batch


def to_x_if(batch, x: str, cond):
    if isinstance(batch, (tuple, list)):
        batch = [to_x(b, x) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_x(v, x) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        if cond(x):
            batch = batch.to(x, non_blocking=True)
    elif isinstance(batch, np.ndarray):  # numpy and others
        if cond(x):
            batch = torch.as_tensor(batch).to(x, non_blocking=True)
    else:
        pass  # do nothing here, used for typed in to_x for methods
        # FIXME: Incosistent behavior here, might lead to undebuggable bugs
    return batch


def to_x(batch, x: str) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)):
        batch = [to_x(b, x) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_x(v, x) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(x, non_blocking=True)
    elif isinstance(batch, np.ndarray):  # numpy and others
        batch = torch.as_tensor(batch).to(x, non_blocking=True)
    else:
        pass  # do nothing here, used for typed in to_x for methods
        # FIXME: Incosistent behavior here, might lead to undebuggable bugs
    return batch


def to_tensor(batch, ignore_list: bool = False) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_tensor(b, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_tensor(v, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        pass
    else:  # numpy and others
        batch = torch.as_tensor(batch)
    return batch


def to_list(batch, non_blocking=False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)):
        batch = [to_list(b, non_blocking) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_list(v, non_blocking) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy().tolist()
    elif isinstance(batch, torch.Tensor):
        batch = batch.tolist()
    else:  # others, keep as is
        pass
    return batch


def to_cpu(batch, non_blocking=False, ignore_list: bool = False) -> torch.Tensor:
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_cpu(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_cpu(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking)
    else:  # numpy and others
        batch = torch.as_tensor(batch, device="cpu")
    return batch


def to_numpy(batch, non_blocking=False, ignore_list: bool = False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_numpy(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_numpy(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.asarray(batch)
    return batch


def remove_batch(batch) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        batch = [remove_batch(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: remove_batch(v) for k, v in batch.items()})
    elif isinstance(batch, (torch.Tensor, np.ndarray)):  # numpy and others
        batch = batch[0]
    else:
        batch = torch.as_tensor(batch)[0]
    return batch


def add_batch(batch) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        batch = [add_batch(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: add_batch(v) for k, v in batch.items()})
    elif isinstance(batch, (torch.Tensor, np.ndarray)):  # numpy and others
        batch = batch[None]
    else:
        batch = torch.as_tensor(batch)[None]
    return batch


def add_iter(batch, iter, total) -> Union[torch.Tensor, np.ndarray]:
    batch = add_scalar(batch, iter, name="iter")
    batch = add_scalar(batch, iter / total, name="frac")
    return batch  # training fraction and current iteration


def add_scalar(batch, value, name) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(batch, (tuple, list)):
        for b in batch:
            add_scalar(b, value, name)

    if isinstance(batch, dict):
        batch[name] = torch.tensor(value)
        batch['meta'][name] = torch.tensor(value)
    return batch


def get_voxel_grid_and_update_bounds(voxel_size: Union[List, np.ndarray], bounds: Union[List, np.ndarray]):
    # now here's the problem
    # 1. if you want the voxel size to be accurate, you bounds need to be changed along with this sampling process
    #    since the F.grid_sample will treat the bounds based on align_corners=True or not
    #    say we align corners, the actual bound on the sampled tpose blend weight should be determined by the actual sampling voxels
    #    not the bound that we kind of used to produce the voxels, THEY DO NOT LINE UP UNLESS your bounds is divisible by the voxel size in every direction
    # TODO: is it possible to somehow get rid of this book-keeping step
    if isinstance(voxel_size, List):
        voxel_size = np.array(voxel_size)
        bounds = np.array(bounds)
    # voxel_size: [0.005, 0.005, 0.005]
    # bounds: n_batch, 2, 3, initial bounds
    x = np.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0] / 2, voxel_size[0])
    y = np.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1] / 2, voxel_size[1])
    z = np.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2] / 2, voxel_size[2])
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).astype(np.float32)
    bounds = np.stack([pts[0, 0, 0], pts[-1, -1, -1]], axis=0).astype(np.float32)
    return pts, bounds


def get_rigid_transform(pose: np.ndarray, joints: np.ndarray, parents: np.ndarray):
    # pose: N, 3
    # joints: N, 3
    # parents: N
    from easyvolcap.utils.blend_utils import get_rigid_transform_nobatch as net_get_rigid_transform
    pose, joints, parents = default_convert([pose, joints, parents])
    J, A = net_get_rigid_transform(pose, joints, parents)
    J, A = to_numpy([J, A])

    return J, A


def get_bounds(xyz, padding=0.05):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= padding
    max_xyz += padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    return bounds


def load_image_file(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg') or img_path.endswith('.JPG') or img_path.endswith('.jpeg') or img_path.endswith('.JPEG'):
        im = Image.open(img_path)
        w, h = im.width, im.height
        draft = im.draft('RGB', (int(w * ratio), int(h * ratio)))
        img = np.asarray(im)
        if np.issubdtype(img.dtype, np.integer):
            img = img.astype(np.float32) / np.iinfo(img.dtype).max  # normalize
        if img.ndim == 2:
            img = img[..., None]
        if ratio != 1.0 and \
            draft is None or \
                draft is not None and \
        (draft[1][2] != int(w * ratio) or
             draft[1][3] != int(h * ratio)):
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
        return img
    else:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.ndim >= 3 and img.shape[-1] >= 3:
            img[..., :3] = img[..., [2, 1, 0]]  # BGR to RGB
        if img.ndim == 2:
            img = img[..., None]
        if np.issubdtype(img.dtype, np.integer):
            img = img.astype(np.float32) / np.iinfo(img.dtype).max  # normalize
        if ratio != 1.0:
            height, width = img.shape[:2]
            img = cv2.resize(img, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
        return img


def load_depth(depth_file: str):
    if depth_file.endswith('.npy'):
        depth = np.load(depth_file)[..., None]  # H, W, 1
    elif depth_file.endswith('.pfm'):
        depth, scale = read_pfm(depth_file)
        depth = depth / scale
        if depth.ndim == 2:
            depth = depth[..., None]  # H, W, 1
        depth = depth[..., :1]
    elif depth_file.endswith('.hdr') or depth_file.endswith('.exr'):
        depth = load_image(depth_file)
        depth = depth[..., :1]
    else:
        raise NotImplementedError
    return depth  # H, W, 1


def load_image(path: Union[str, np.ndarray], ratio: int = 1.0):
    if isinstance(path, str):
        return load_image_file(path, ratio)
    elif isinstance(path, np.ndarray):
        return load_image_from_bytes(path, ratio)
    else:
        raise NotImplementedError('Supported overloading')


def load_unchanged(img_path: str, ratio=1.0):
    if img_path.endswith('.jpg') or img_path.endswith('.JPG') or img_path.endswith('.jpeg') or img_path.endswith('.JPEG'):
        im = Image.open(img_path)
        w, h = im.width, im.height
        draft = im.draft('RGB', (int(w * ratio), int(h * ratio)))
        img = np.asarray(im).copy()  # avoid writing error and already in RGB instead of BGR
        if ratio != 1.0 and \
            draft is None or \
                draft is not None and \
        (draft[1][2] != int(w * ratio) or
             draft[1][3] != int(h * ratio)):
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
        return img
    else:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] >= 3:
            img[..., :3] = img[..., [2, 1, 0]]
        if ratio != 1.0:
            height, width = img.shape[:2]
            img = cv2.resize(img, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
        return img


def load_mask(msk_path: str, ratio=1.0):
    if msk_path.endswith('.jpg') or msk_path.endswith('.JPG') or msk_path.endswith('.jpeg') or msk_path.endswith('.JPEG'):
        msk = Image.open(msk_path)
        w, h = msk.width, msk.height
        draft = msk.draft('L', (int(w * ratio), int(h * ratio)))
        msk = np.asarray(msk).astype(int)  # read the actual file content from drafted disk
        msk = msk * 255 / msk.max()  # if max already 255, do nothing
        # msk = msk[..., None] > 128
        msk = msk.astype(np.uint8)
        if ratio != 1.0 and \
            draft is None or \
                draft is not None and \
        (draft[1][2] != int(w * ratio) or
             draft[1][3] != int(h * ratio)):
            msk = cv2.resize(msk.astype(np.uint8), (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)[..., None]
        return msk
    else:
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE).astype(int)  # BGR to GRAY
        msk = msk * 255 / msk.max()  # if max already 255, do nothing
        # msk = msk[..., None] > 128  # make it binary
        msk = msk.astype(np.uint8)
        if ratio != 1.0:
            height, width = msk.shape[:2]
            msk = cv2.resize(msk.astype(np.uint8), (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_NEAREST)[..., None]
            # WTF: https://stackoverflow.com/questions/68502581/image-channel-missing-after-resizing-image-with-opencv
        return msk


def save_unchanged(img_path: str, img: np.ndarray, quality=100, compression=6):
    if img.shape[-1] >= 3:
        img[..., :3] = img[..., [2, 1, 0]]
    if img_path.endswith('.hdr'):
        return cv2.imwrite(img_path, img)  # nothing to say about hdr
    if dirname(img_path):
        os.makedirs(dirname(img_path), exist_ok=True)
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_PNG_COMPRESSION, compression])


def save_image(img_path: str, img: np.ndarray, jpeg_quality=75, png_compression=9, save_dtype=np.uint8):
    if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()
    if img.ndim == 4: img = np.concatenate(img, axis=0)
    if img.shape[0] < img.shape[-1] and (img.shape[0] == 3 or img.shape[0] == 4): img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.integer):
        img = img / np.iinfo(img.dtype).max  # to float
    if img.shape[-1] >= 3:
        if not img.flags['WRITEABLE']:
            img = img.copy()  # avoid assignment only inputs
        img[..., :3] = img[..., [2, 1, 0]]
    if dirname(img_path):
        os.makedirs(dirname(img_path), exist_ok=True)
    if img_path.endswith('.png'):
        max = np.iinfo(save_dtype).max
        img = (img * max).clip(0, max).astype(save_dtype)
    elif img_path.endswith('.jpg'):
        img = img[..., :3]  # only color
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img_path.endswith('.hdr'):
        img = img[..., :3]  # only color
    elif img_path.endswith('.exr'):
        # ... https://github.com/opencv/opencv/issues/21326
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    else:
        # should we try to discard alpha channel here?
        # exr could store alpha channel
        pass  # no transformation for other unspecified file formats
    return cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                                       cv2.IMWRITE_PNG_COMPRESSION, png_compression,
                                       cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PIZ])


def save_mask(msk_path: str, msk: np.ndarray, quality=75, compression=9):
    if dirname(msk_path):
        os.makedirs(dirname(msk_path), exist_ok=True)
    if msk.ndim == 2:
        msk = msk[..., None]
    return cv2.imwrite(msk_path, msk[..., 0] * 255, [cv2.IMWRITE_JPEG_QUALITY, quality,
                                                     cv2.IMWRITE_PNG_COMPRESSION, compression,
                                                     cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PIZ])


def list_to_numpy(x: list): return np.stack(x).transpose(0, 3, 1, 2)


def numpy_to_list(x: np.ndarray): return [y for y in x.transpose(0, 2, 3, 1)]


def list_to_tensor(x: list, device='cuda'): return torch.from_numpy(list_to_numpy(x)).to(device, non_blocking=True)  # convert list of numpy arrays of HWC to BCHW


def tensor_to_list(x: torch.Tensor): return numpy_to_list(x.detach().cpu().numpy())  # convert tensor of BCHW to list of numpy arrays of HWC


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def read_mask_by_img_path(data_root: str, img_path: str, erode_dilate_edge: bool = False, mask: str = '') -> np.ndarray:
    def read_mask_file(path):
        msk = load_mask(path).astype(np.uint8)
        if len(msk.shape) == 3:
            msk = msk[..., 0]
        return msk

    if mask:
        msk_path = os.path.join(data_root, img_path.replace('images', mask))
        if not os.path.exists(msk_path):
            msk_path = os.path.join(data_root, img_path.replace('images', mask)) + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(data_root, img_path.replace('images', mask))[:-4] + '.png'
        if not os.path.exists(msk_path):
            log(f'warning: defined mask path {msk_path} does not exist', 'yellow')
    else:
        msk_path = os.path.join(data_root, 'mask', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, 'mask', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, 'mask_cihp', img_path)[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'merged_mask'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'rvm'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'rvm'))[:-4] + '.jpg'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'mask'))[:-4] + '.png'
    if not os.path.exists(msk_path):  # background matte v2
        msk_path = os.path.join(data_root, img_path.replace('images', 'bgmt'))[:-4] + '.png'
    if not os.path.exists(msk_path):
        msk_path = os.path.join(data_root, img_path.replace('images', 'mask'))[:-4] + '.jpg'
    if not os.path.exists(msk_path):
        log(f'cannot find mask file: {msk_path}, using all ones', 'yellow')
        img = load_unchanged_image(os.path.join(data_root, img_path))
        msk = np.ones_like(img[:, :, 0]).astype(np.uint8)
        return msk

    msk = read_mask_file(msk_path)
    # erode edge inconsistence when evaluating and training
    if erode_dilate_edge:  # eroding edge on matte might erode the actual human
        msk = fill_mask_edge_with(msk)

    return msk


def fill_mask_edge_with(msk, border=5, value=100):
    msk = msk.copy()
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = value
    return msk


def get_rays_within_bounds_rendering(H, W, K, R, T, bounds):
    ray_o, ray_d = get_rays(H, W, K, R, T)

    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    near = near.reshape(H, W)
    far = far.reshape(H, W)
    ray_o = ray_o.reshape(H, W, 3)
    ray_d = ray_d.reshape(H, W, 3)
    mask_at_box = mask_at_box.reshape(H, W)

    return ray_o, ray_d, near, far, mask_at_box


def get_rays(H, W, K, R, T):
    # # calculate the camera origin
    # ray_o = -np.dot(R.T, T).ravel()
    # # calculate the world coodinates of pixels
    # i, j = np.meshgrid(np.arange(H, dtype=np.float32),
    #                    np.arange(W, dtype=np.float32),
    #                    indexing='ij')  # 0.5 indicates pixel center
    # i = i + 0.5
    # j = j + 0.5
    # # 0->H, 0->W
    # xy1 = np.stack([j, i, np.ones_like(i)], axis=2)
    # if subpixel:
    #     rand = np.random.rand(H, W, 2) - 0.5
    #     xy1[:, :, :2] += rand
    # pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    # pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # # calculate the ray direction
    # ray_d = pixel_world - ray_o[None, None]
    # ray_d = ray_d / np.linalg.norm(ray_d, axis=2, keepdims=True)
    # ray_o = np.broadcast_to(ray_o, ray_d.shape)
    # return ray_o, ray_d

    from easyvolcap.utils.ray_utils import get_rays
    K, R, T = to_tensor([K, R, T])
    ray_o, ray_d = get_rays(H, W, K, R, T)
    ray_o, ray_d = to_numpy([ray_o, ray_d])
    return ray_o, ray_d


def get_near_far(bounds, ray_o, ray_d) -> Tuple[np.ndarray, np.ndarray]:
    # """
    # calculate intersections with 3d bounding box
    # return: near, far (indexed by mask_at_box (bounding box mask))
    # """
    # near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    # norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    # near = near[mask_at_box] / norm_d[mask_at_box, 0]
    # far = far[mask_at_box] / norm_d[mask_at_box, 0]
    # return near, far, mask_at_box
    from easyvolcap.utils.ray_utils import get_near_far_aabb
    bounds, ray_o, ray_d = to_tensor([bounds, ray_o, ray_d])  # no copy
    near, far = get_near_far_aabb(bounds, ray_o, ray_d)
    near, far = to_numpy([near, far])
    return near, far


def get_full_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near / norm_d[..., 0]
    far = far / norm_d[..., 0]
    return near, far, mask_at_box


def full_sample_ray(img, msk, K, R, T, bounds, split='train', subpixel=False):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T, subpixel)
    near, far, mask_at_box = get_full_near_far(bounds, ray_o, ray_d)
    msk = msk * mask_at_box
    coords = np.argwhere(np.ones_like(mask_at_box))  # every pixel
    ray_o = ray_o[coords[:, 0], coords[:, 1]].astype(np.float32)
    ray_d = ray_d[coords[:, 0], coords[:, 1]].astype(np.float32)
    near = near[coords[:, 0], coords[:, 1]].astype(np.float32)
    far = far[coords[:, 0], coords[:, 1]].astype(np.float32)
    rgb = img[coords[:, 0], coords[:, 1]].astype(np.float32)
    return rgb, ray_o, ray_d, near, far, coords, mask_at_box


def affine_inverse(m: np.ndarray):
    import torch
    from easyvolcap.utils.math_utils import affine_inverse
    return affine_inverse(torch.from_numpy(m)).numpy()


def load_image_from_bytes(buffer: np.ndarray, ratio=1.0, normalize=False, decode_flag=cv2.IMREAD_UNCHANGED):
    # from nvjpeg import NvJpeg
    # if not hasattr(load_image_from_bytes, 'nj'):
    #     load_image_from_bytes.nj = NvJpeg()
    # nj: NvJpeg = load_image_from_bytes.nj

    def normalize_image(image):
        image = torch.from_numpy(image)  # pytorch is significantly faster than np
        if image.ndim >= 3 and image.shape[-1] >= 3:
            image[..., :3] = image[..., [2, 1, 0]]
        image = image / torch.iinfo(image.dtype).max
        image = image.float()
        return image.numpy()

    if isinstance(buffer, BytesIO):
        buffer = buffer.getvalue()  # slow? copy?
    if isinstance(buffer, memoryview) or isinstance(buffer, bytes):
        buffer = np.frombuffer(buffer, np.uint8)
    if isinstance(buffer, torch.Tensor):
        buffer = buffer.numpy()
    buffer = buffer.astype(np.uint8)
    image: np.ndarray = cv2.imdecode(buffer, decode_flag)  # MARK: 10-15ms
    # image: np.ndarray = nj.decode(np.frombuffer(buffer, np.uint8))  # MARK: 10-15ms
    # if decode_flag == cv2.IMREAD_GRAYSCALE:
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim == 2:
        image = image[..., None]

    if normalize:
        image = normalize_image(image)  # MARK: 3ms

    height, width = image.shape[:2]
    if ratio != 1.0:
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
    return image


def as_torch_func(func):
    def wrapper(*args, **kwargs):
        args = to_numpy(args)
        kwargs = to_numpy(kwargs)
        ret = func(*args, **kwargs)
        return to_tensor(ret)
    return wrapper


def as_numpy_func(func):
    def wrapper(*args, **kwargs):
        args = to_tensor(args)
        kwargs = to_tensor(kwargs)
        ret = func(*args, **kwargs)
        return to_numpy(ret)
    return wrapper


def load_image_bytes(im: str):
    if im.endswith('.exr'):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    with open(im, "rb") as fh:
        buffer = fh.read()
    return buffer


class UnstructuredTensors(torch.Tensor):
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519
    # https://github.com/pytorch/pytorch/issues/69893
    @staticmethod
    def __new__(cls, bytes: Union[List[np.ndarray], List[torch.Tensor], np.ndarray], **kwargs):
        """
        Creates a new UnstructuredTensors object from the given bytes.

        Args:
        - bytes (Union[List[np.ndarray], List[torch.Tensor], np.ndarray]): The bytes to create the object from.

        Returns:
        - self (UnstructuredTensors): The new UnstructuredTensors object.
        """
        if isinstance(bytes, UnstructuredTensors):
            return bytes
        # Prepare the bytes array
        if isinstance(bytes, np.ndarray):
            bytes = [b for b in bytes]
        if bytes[0].dtype == object:
            bytes = [b.astype(np.uint8) for b in bytes]
        bytes = to_tensor(bytes)  # now, every element is a list
        dtype = torch.uint8
        if len(bytes):
            dtype = bytes[0].dtype

        # Create an empty tensor
        self = torch.Tensor.__new__(cls).to(dtype)

        # Remember accessing related configs
        self.set_(torch.cat(bytes))  # flatten # sum(N)
        self.lengths = torch.as_tensor([len(b) for b in bytes], dtype=torch.int32)  # N,
        self.cumsums = torch.cat([torch.as_tensor([0]), torch.cumsum(self.lengths, dim=0)[:-1]])
        return self

    @property
    def is_unstructured(self): return hasattr(self, 'lengths')

    def __getitem__(self, index: int):
        """
        Returns a slice of the UnstructuredTensors object corresponding to the given index.

        Args:
            index (int): The index of the slice to return.

        Returns:
            torch.Tensor: A slice of the UnstructuredTensors object corresponding to the given index.

        This function returns a slice of the UnstructuredTensors object corresponding to the given index. The slice is obtained by using the cumulative sums and lengths of the underlying bytes array to determine the start and end indices of the slice. If the index is out of range, the function returns the corresponding element of the underlying bytes array. This function is used to implement the indexing behavior of the UnstructuredTensors object, allowing it to be treated like a regular tensor.
        """
        if self.is_unstructured:
            return torch.Tensor.__getitem__(self, slice(self.cumsums[index], self.cumsums[index] + self.lengths[index]))
        else:
            return super().__getitem__(index)

    def __len__(self):
        if self.is_unstructured:
            return len(self.lengths)
        else:
            return super().__len__()

    def clone(self, *args, **kwargs):
        if self.is_unstructured:
            return UnstructuredTensors([self[i] for i in range(len(self.lengths))])  # manual cloning with copy and reconstruction
        else:
            return super().clone(*args, **kwargs)


def load_ims_bytes_from_disk(ims: np.ndarray, desc="Loading image bytes from disk"):
    sh = ims.shape
    ims = ims.ravel()
    ims_bytes = parallel_execution(list(ims), action=load_image_bytes, desc=desc, print_progress=True)
    ims_bytes = np.asarray(ims_bytes).reshape(sh)  # reorganize shapes
    return ims_bytes


def load_resize_undist_im_bytes(imp: str,
                                K: np.ndarray,
                                D: np.ndarray,
                                ratio: Union[float, List[int]] = 1.0,
                                center_crop_size: List[int] = [-1, -1],
                                encode_ext='.jpg',
                                decode_flag=cv2.IMREAD_UNCHANGED,
                                dist_opt_K: bool = False,
                                jpeg_quality: int = 100,
                                png_compression: int = 6
                                ):
    # Load image -> resize -> undistort -> save to bytes (jpeg)
    img = load_image_from_bytes(load_image_bytes(imp), decode_flag=decode_flag)[..., :3]  # cv2 decoding (fast)

    oH, oW = img.shape[:2]

    if dist_opt_K:
        newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(K, D, (oW, oH), 0, (oW, oH))
        img = cv2.undistort(img, K, D, newCameraMatrix=newCameraMatrix)
        K = newCameraMatrix
    else:
        img = cv2.undistort(img, K, D)

    # Maybe update image size
    if not ((isinstance(ratio, float) and ratio == 1.0)):
        if isinstance(ratio, float):
            H, W = int(oH * ratio), int(oW * ratio)
        else:
            H, W = ratio  # ratio is actually the target image size
        rH, rW = H / oH, W / oW
        K = K.copy()
        K[0:1] = K[0:1] * rW  # K[0, 0] *= rW
        K[1:2] = K[1:2] * rH  # K[1, 1] *= rH

        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # H, W, 3, uint8

    # Crop the image and intrinsic matrix if specified
    if center_crop_size[0] > 0:
        img, K, H, W = center_crop_img_ixt(img, K, H, W, center_crop_size)

    is_success, buffer = cv2.imencode(encode_ext, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_PNG_COMPRESSION, png_compression])

    if 'H' not in locals(): H, W = oH, oW
    return buffer, K, H, W


def center_crop_img_ixt(img: np.ndarray, K: np.ndarray, H: int, W: int,
                        center_crop_size: Union[int, List[int]]):
    # Parse the original size and the target crop size
    oH, oW = H, W
    if isinstance(center_crop_size, int): cH, cW = center_crop_size, center_crop_size
    else: cH, cW = center_crop_size

    # Compute left and right crop size for height and width respectively
    hlc, wlc = int((oH - cH) * 0.5), int((oW - cW) * 0.5)
    hrc, wrc = oH - cH - hlc, oW - cW - wlc

    # Crop the image
    if hlc != 0: img = img[hlc:-hrc, :]
    if wlc != 0: img = img[:, wlc:-wrc]

    # Crop the intrinsic matrix
    if hlc != 0: K[1, 2] -= hlc
    if wlc != 0: K[0, 2] -= wlc

    return img, K, cH, cW


def load_resize_undist_ims_bytes(ims: np.ndarray,
                                 Ks: np.ndarray,
                                 Ds: np.ndarray,
                                 ratio: Union[float, List[int], List[float]] = 1.0,
                                 center_crop_size: List[int] = [-1, -1],
                                 desc="Loading image bytes from disk",
                                 **kwargs):
    sh = ims.shape  # V, N
    # Ks = np.broadcast_to(Ks[:, None], (sh + (3, 3)))
    # Ds = np.broadcast_to(Ds[:, None], (sh + (1, 5)))

    ims = ims.reshape((np.prod(sh)))
    # from easyvolcap.utils.dist_utils import get_rank
    # if not get_rank(): __import__('easyvolcap.utils.console_utils', fromlist=['debugger']).debugger()
    # else:
    #     while 1: pass
    Ks = Ks.reshape((np.prod(sh), 3, 3))
    Ds = Ds.reshape((np.prod(sh), 1, 5))

    ims = list(ims)
    Ks = list(Ks)
    Ds = list(Ds)  # only convert outer most dim to list

    if isinstance(ratio, list) and len(ratio) and isinstance(ratio[0], float):
        ratio = np.broadcast_to(np.asarray(ratio)[:, None], sh)  # V, N
        ratio = ratio.reshape((np.prod(sh)))
        ratio = list(ratio)
    elif isinstance(ratio, list):
        ratio = np.asarray(ratio)  # avoid expansion in parallel execution

    if isinstance(center_crop_size, list):
        center_crop_size = np.asarray(center_crop_size)  # avoid expansion

    # Should we batch these instead of loading?
    out = parallel_execution(ims, Ks, Ds, ratio, center_crop_size,
                             action=load_resize_undist_im_bytes,
                             desc=desc, print_progress=True,
                             **kwargs,
                             )

    ims_bytes, Ks, Hs, Ws = zip(*out)  # is this OK?
    ims_bytes, Ks, Hs, Ws = np.asarray(ims_bytes, dtype=object), np.asarray(Ks), np.asarray(Hs), np.asarray(Ws)
    # ims_bytes = ims_bytes.reshape(sh)  # numpy array of bytesio
    Hs = Hs.reshape(sh)  # should all be the same?
    Ws = Ws.reshape(sh)  # should all be the same?
    Ks = Ks.reshape(sh + (3, 3))  # should all be the same?

    return ims_bytes, Ks, Hs, Ws


def decode_crop_fill_im_bytes(im_bytes: BytesIO,
                              mk_bytes: BytesIO,
                              K: np.ndarray,
                              R: np.ndarray,
                              T: np.ndarray,
                              bounds: np.ndarray,
                              encode_ext=['.jpg', '.jpg'],
                              decode_flag=cv2.IMREAD_UNCHANGED,
                              jpeg_quality: int = 100,
                              png_compression: int = 6,
                              **kwargs):
    # im_bytes: a series of jpeg bytes for the image
    # mk_bytes: a series of jpeg bytes for the mask
    # K: 3, 3 intrinsics matrix

    # Use load_image_from_bytes to decode and update jpeg streams
    img = load_image_from_bytes(im_bytes, decode_flag=decode_flag)  # H, W, 3
    msk = load_image_from_bytes(mk_bytes, decode_flag=decode_flag)  # H, W, 3

    # Crop both mask and the image using bbox's 2D projection
    H, W, _ = img.shape
    from easyvolcap.utils.bound_utils import get_bound_2d_bound
    bx, by, bw, bh = as_numpy_func(get_bound_2d_bound)(bounds, K, R, T, H, W)
    img = img[by:by + bh, bx:bx + bw]
    msk = msk[by:by + bh, bx:bx + bw]

    # Crop the image using the bounding rect of the mask
    mx, my, mw, mh = cv2.boundingRect((msk > 128).astype(np.uint8))  # array data type = 0 is not supported
    img = img[my:my + mh, mx:mx + mw]
    msk = msk[my:my + mh, mx:mx + mw]

    # Update the final size and intrinsics
    x, y, w, h = bx + mx, by + my, mw, mh  # w and h will always be the smaller one, xy will be accumulated
    K[0, 2] -= x
    K[1, 2] -= y

    # Fill the image with black (premultiply by mask)
    img = (img * (msk / 255)).clip(0, 255).astype(np.uint8)  # fill with black, indexing starts at the front

    # Reencode the videos and masks
    if isinstance(encode_ext, str): encode_ext = [encode_ext] * 2  # '.jpg' -> ['.jpg', '.jpg']
    im_bytes = cv2.imencode(encode_ext[0], img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1]  # is_sucess, bytes_array
    mk_bytes = cv2.imencode(encode_ext[1], msk, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1]  # is_sucess, bytes_array
    return im_bytes, mk_bytes, K, h, w, x, y


def decode_crop_fill_ims_bytes(ims_bytes: np.ndarray, mks_bytes: np.ndarray, Ks: np.ndarray, Rs: np.ndarray, Ts: np.ndarray, bounds: np.ndarray,
                               desc="Cropping images using mask", **kwargs):
    sh = Ks.shape[:2]  # V, N
    # ims_bytes = ims_bytes.reshape((np.prod(sh)))
    # mks_bytes = mks_bytes.reshape((np.prod(sh)))
    Ks = Ks.reshape((np.prod(sh), 3, 3))
    Rs = Rs.reshape((np.prod(sh), 3, 3))
    Ts = Ts.reshape((np.prod(sh), 3, 1))
    bounds = bounds.reshape((np.prod(sh), 2, 3))

    # Should we batch these instead of loading?
    out = parallel_execution(list(ims_bytes), list(mks_bytes), list(Ks), list(Rs), list(Ts), list(bounds),
                             action=decode_crop_fill_im_bytes,
                             desc=desc, print_progress=True,
                             **kwargs,
                             )

    ims_bytes, mks_bytes, Ks, Hs, Ws, xs, ys = zip(*out)  # is this OK?
    ims_bytes, mks_bytes, Ks, Hs, Ws, xs, ys = np.asarray(ims_bytes, dtype=object), np.asarray(mks_bytes, dtype=object), np.asarray(Ks), np.asarray(Hs), np.asarray(Ws), np.asarray(xs), np.asarray(ys)
    # ims_bytes = ims_bytes.reshape(sh)
    # mks_bytes = mks_bytes.reshape(sh)
    Hs = Hs.reshape(sh)  # should all be the same?
    Ws = Ws.reshape(sh)  # should all be the same?
    Ks = Ks.reshape(sh + (3, 3))  # should all be the same?
    xs = xs.reshape(sh)  # should all be the same?
    ys = ys.reshape(sh)  # should all be the same?

    return ims_bytes, mks_bytes, Ks, Hs, Ws, xs, ys


def decode_fill_im_bytes(im_bytes: BytesIO,
                         mk_bytes: BytesIO,
                         encode_ext='.jpg',
                         decode_flag=cv2.IMREAD_UNCHANGED,
                         jpeg_quality: int = 100,
                         png_compression: int = 6,
                         **kwargs):
    # im_bytes: a series of jpeg bytes for the image
    # mk_bytes: a series of jpeg bytes for the mask
    # K: 3, 3 intrinsics matrix

    # Use load_image_from_bytes to decode and update jpeg streams
    img = load_image_from_bytes(im_bytes, decode_flag=decode_flag)  # H, W, 3
    msk = load_image_from_bytes(mk_bytes, decode_flag=decode_flag)  # H, W, 3

    img = (img * (msk / 255)).clip(0, 255).astype(np.uint8)  # fill with black, indexing starts at the front

    im_bytes = cv2.imencode(encode_ext, img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_PNG_COMPRESSION, png_compression])[1]  # is_sucess, bytes_array
    return im_bytes


def decode_fill_ims_bytes(ims_bytes: np.ndarray,
                          mks_bytes: np.ndarray,
                          desc="Filling images using mask",
                          **kwargs):
    sh = ims_bytes.shape  # V, N
    ims_bytes = ims_bytes.reshape((np.prod(sh)))
    mks_bytes = mks_bytes.reshape((np.prod(sh)))

    # Should we batch these instead of loading?
    ims_bytes = parallel_execution(list(ims_bytes), list(mks_bytes),
                                   action=decode_fill_im_bytes,
                                   desc=desc, print_progress=True,
                                   **kwargs,
                                   )

    ims_bytes = np.asarray(ims_bytes, dtype=object)
    ims_bytes = ims_bytes.reshape(sh)
    return ims_bytes


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat.astype(np.float32)


def get_rigid_transformation_and_joints(poses, joints, parents):
    """
    poses: n_bones x 3
    joints: n_bones x 3
    parents: n_bones
    """

    n_bones = len(joints)
    rot_mats = batch_rodrigues(poses)

    # Obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # Create the transformation matrix
    # First rotate then transform
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([n_bones, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # Rotate each part
    # But this is a world transformation, with displacement...?
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):  # assuming parents are in topological order
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])  # THEY'RE RIGHT, LEARN FORWARD KINEMATICS
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # Obtain the rigid transformation
    # AND THIS WEIRD STUFF IS TRYING TO MOVE VERTEX FROM VERTEX COORDINATES TO JOINT COORDINATES
    # AND THIS IS THE CORRECT IMPLEMENTATION...

    # THIS IS JUST TOO CLEVER...
    # These three lines is effectively doing: transforms = transforms * (negative trarslation matrix for all joints)
    joints_vector = np.concatenate([joints, np.zeros([n_bones, 1])], axis=1)
    rot_joints = np.sum(transforms * joints_vector[:, None], axis=2)  # This is effectively matmul
    transforms[..., 3] = transforms[..., 3] - rot_joints  # add in the translation, we should translate first

    joints_points = np.concatenate([joints, np.ones([n_bones, 1])], axis=1)
    pose_joints = np.sum(transforms * joints_points[:, None], axis=2)  # This is effectively matmul

    transforms = transforms.astype(np.float32)
    return transforms, pose_joints[:, :3]


def get_rigid_transformation(poses, joints, parents):
    """
    poses: n_bones x 3
    joints: n_bones x 3
    parents: n_bones
    """
    transforms = get_rigid_transformation_and_joints(poses, joints, parents)[0]
    return transforms


def padding_bbox_HW(bbox, h, w):
    padding = 10
    bbox[0] = bbox[0] - 10
    bbox[1] = bbox[1] + 10

    height = bbox[1, 1] - bbox[0, 1]
    width = bbox[1, 0] - bbox[0, 0]
    # a magic number of pytorch3d
    ratio = 1.5

    if height / width > ratio:
        min_size = int(height / ratio)
        if width < min_size:
            padding = (min_size - width) // 2
            bbox[0, 0] = bbox[0, 0] - padding
            bbox[1, 0] = bbox[1, 0] + padding

    if width / height > ratio:
        min_size = int(width / ratio)
        if height < min_size:
            padding = (min_size - height) // 2
            bbox[0, 1] = bbox[0, 1] - padding
            bbox[1, 1] = bbox[1, 1] + padding

    bbox[:, 0] = np.clip(bbox[:, 0], a_min=0, a_max=w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], a_min=0, a_max=h - 1)

    return bbox


def padding_bbox(bbox, img):
    return padding_bbox_HW(bbox, *img.shape[:2])


def get_crop_box(H, W, K, ref_msk):
    x, y, w, h = cv2.boundingRect(ref_msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox_HW(bbox, H, W)

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return K, bbox


def crop_image_msk(img, msk, K, ref_msk):
    x, y, w, h = cv2.boundingRect(ref_msk)
    bbox = np.array([[x, y], [x + w, y + h]])
    bbox = padding_bbox(bbox, img)

    crop = img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
    crop_msk = msk[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]

    # calculate the shape
    shape = crop.shape
    x = 8
    height = (crop.shape[0] | (x - 1)) + 1
    width = (crop.shape[1] | (x - 1)) + 1

    # align image
    aligned_image = np.zeros([height, width, 3])
    aligned_image[:shape[0], :shape[1]] = crop
    aligned_image = aligned_image.astype(np.float32)

    # align mask
    aligned_msk = np.zeros([height, width])
    aligned_msk[:shape[0], :shape[1]] = crop_msk
    aligned_msk = (aligned_msk == 1).astype(np.uint8)

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - bbox[0, 0]
    K[1, 2] = K[1, 2] - bbox[0, 1]
    K = K.astype(np.float32)

    return aligned_image, aligned_msk, K, bbox


def random_crop_image(img, msk, K, min_size, max_size):
    # sometimes we sample regions with no valid pixel at all, this can be problematic for the training loop
    # there's an assumption that the `msk` is always inside `mask_at_box`
    # thus, if we're sampling inside the `msk`, we'll always be getting the correct results
    H, W = img.shape[:2]
    min_HW = min(H, W)
    min_HW = min(min_HW, max_size)

    max_size = min_HW
    # min_size = int(min(min_size, 0.8 * min_HW))
    if max_size < min_size:
        H_size = np.random.randint(min_size, max_size)
    else:
        H_size = min_size

    W_size = H_size
    x = 8
    H_size = (H_size | (x - 1)) + 1
    W_size = (W_size | (x - 1)) + 1

    # randomly select begin_x and begin_y
    coords = np.argwhere(msk == 1)
    center_xy = coords[np.random.randint(0, len(coords))][[1, 0]]
    min_x, min_y = center_xy[0] - W_size // 2, center_xy[1] - H_size // 2
    max_x, max_y = min_x + W_size, min_y + H_size
    if min_x < 0:
        min_x, max_x = 0, W_size
    if max_x > W:
        min_x, max_x = W - W_size, W
    if min_y < 0:
        min_y, max_y = 0, H_size
    if max_y > H:
        min_y, max_y = H - H_size, H

    # crop image and mask
    begin_x, begin_y = min_x, min_y
    img = img[begin_y:begin_y + H_size, begin_x:begin_x + W_size]
    msk = msk[begin_y:begin_y + H_size, begin_x:begin_x + W_size]

    # revise the intrinsic camera matrix
    K = K.copy()
    K[0, 2] = K[0, 2] - begin_x
    K[1, 2] = K[1, 2] - begin_y
    K = K.astype(np.float32)

    return img, msk, K


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.asarray([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ], dtype=np.float32)
    return corners_3d


def get_bound_2d_mask(bounds, K, RT, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, RT)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask


def get_bounds(xyz, box_padding=0.05):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= box_padding
    max_xyz += box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds


def crop_mask_edge(msk):
    msk = msk.copy()
    border = 10
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = 100
    return msk


def adjust_hsv(img, saturation, brightness, contrast):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[..., 1] = hsv[..., 1] * saturation
    hsv[..., 1] = np.minimum(hsv[..., 1], 255)
    hsv[..., 2] = hsv[..., 2] * brightness
    hsv[..., 2] = np.minimum(hsv[..., 2], 255)
    hsv = hsv.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = img.astype(np.float32) * contrast
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img
