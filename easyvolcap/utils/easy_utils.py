# easymocap utility functions
import os
import cv2
import torch
import numpy as np
import pkg_resources
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import to_numpy


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out + '\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.10f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'real':
            if isinstance(value, np.ndarray):
                value = value.item()
            self._write('{}: {:.10f}'.format(key, value))  # as accurate as possible
        else:
            raise NotImplementedError

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'real':
            output = self.fs.getNode(key).real()
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera(intri_path: str, extri_path: str = None, cam_names=[]) -> dotdict:
    if extri_path is None:
        extri_path = join(intri_path, 'extri.yml')
        intri_path = join(intri_path, 'intri.yml')
    assert exists(intri_path), intri_path
    assert exists(extri_path), extri_path

    intri = FileStorage(intri_path)
    extri = FileStorage(extri_path)
    cams = dotdict()
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # Intrinsics
        cams[cam] = dotdict()
        cams[cam].K = intri.read('K_{}'.format(cam))
        cams[cam].H = int(intri.read('H_{}'.format(cam), dt='real')) or -1
        cams[cam].W = int(intri.read('W_{}'.format(cam), dt='real')) or -1
        cams[cam].invK = np.linalg.inv(cams[cam]['K'])

        # Extrinsics
        Tvec = extri.read('T_{}'.format(cam))
        Rvec = extri.read('R_{}'.format(cam))
        if Rvec is not None: R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read('Rot_{}'.format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        cams[cam].R = R
        cams[cam].T = Tvec
        cams[cam].C = - Rvec.T @ Tvec
        cams[cam].RT = RT
        cams[cam].Rvec = Rvec
        cams[cam].P = cams[cam].K @ cams[cam].RT

        # Distortion
        D = intri.read('D_{}'.format(cam))
        if D is None: D = intri.read('dist_{}'.format(cam))
        cams[cam].D = D

        # Time input
        cams[cam].t = extri.read('t_{}'.format(cam), dt='real') or 0  # temporal index, might all be 0
        cams[cam].v = extri.read('v_{}'.format(cam), dt='real') or 0  # temporal index, might all be 0

        # Bounds, could be overwritten
        cams[cam].n = extri.read('n_{}'.format(cam), dt='real') or 0.0001  # temporal index, might all be 0
        cams[cam].f = extri.read('f_{}'.format(cam), dt='real') or 1e6  # temporal index, might all be 0
        cams[cam].bounds = extri.read('bounds_{}'.format(cam))
        cams[cam].bounds = np.array([[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]]) if cams[cam].bounds is None else cams[cam].bounds

        # CCM
        cams[cam].ccm = intri.read('ccm_{}'.format(cam))
        cams[cam].ccm = np.eye(3) if cams[cam].ccm is None else cams[cam].ccm

    # # Average
    # avg_c2w_R = extri.read('avg_c2w_R')
    # avg_c2w_T = extri.read('avg_c2w_T')
    # if avg_c2w_R is not None: cams.avg_c2w_R = avg_c2w_R
    # if avg_c2w_T is not None: cams.avg_c2w_T = avg_c2w_T

    return dotdict(cams)


def write_camera(cameras: dict, path: str, intri_name: str = '', extri_name: str = ''):
    os.makedirs(path, exist_ok=True)
    if not intri_name or not extri_name:
        intri_name = join(path, 'intri.yml')  # TODO: make them arguments
        extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    cam_names = [key_.split('.')[0] for key_ in cameras.keys()]
    intri.write('names', cam_names, 'list')
    extri.write('names', cam_names, 'list')

    cameras = dotdict(cameras)
    for key_, val in cameras.items():
        # Skip special keys
        if key_ == 'basenames': continue
        # if key_ == 'avg_R': continue
        # if key_ == 'avg_T': continue

        key = key_.split('.')[0]
        # Intrinsics
        intri.write('K_{}'.format(key), val.K)
        if 'H' in val: intri.write('H_{}'.format(key), val.H, 'real')
        if 'W' in val: intri.write('W_{}'.format(key), val.W, 'real')

        # Distortion
        if 'D' not in val:
            if 'dist' in val: val.D = val.dist
            else: val.D = np.zeros((5, 1))
        intri.write('D_{}'.format(key), val.D.reshape(5, 1))

        # Extrinsics
        if 'R' not in val: val.R = cv2.Rodrigues(val.Rvec)[0]
        if 'Rvec' not in val: val.Rvec = cv2.Rodrigues(val.R)[0]
        extri.write('R_{}'.format(key), val.Rvec)
        extri.write('Rot_{}'.format(key), val.R)
        extri.write('T_{}'.format(key), val.T.reshape(3, 1))

        # Temporal
        if 't' in val: extri.write('t_{}'.format(key), val.t, 'real')

        # Bounds
        if 'n' in val: extri.write('n_{}'.format(key), val.n, 'real')
        if 'f' in val: extri.write('f_{}'.format(key), val.f, 'real')
        if 'bounds' in val: extri.write('bounds_{}'.format(key), val.bounds)

        # Color correction matrix
        if 'ccm' in val: intri.write('ccm_{}'.format(key), val.ccm)

    # # Averaged camera matrix (optional)
    # if 'c2w_avg' in cameras:
    #     cameras.avg_R = cameras.c2w_avg[:3, :3]
    #     cameras.avg_T = cameras.c2w_avg[:3, 3:]
    # if 'avg_R' in cameras and 'avg_T' in cameras:
    #     extri.write('avg_R'.format(key), cameras.avg_R)
    #     extri.write('avg_T'.format(key), cameras.avg_T.reshape(3, 1))


def to_easymocap(Ks: torch.Tensor, Hs: torch.Tensor, Ws: torch.Tensor,
                 Rs: torch.Tensor, Ts: torch.Tensor, ts: torch.Tensor,
                 ns: torch.Tensor, fs: torch.Tensor, Ds: torch.Tensor = None,
                 cam_digit: int = 6):
    # Number of render views
    n_render_views = Ks.shape[0]

    # Convert interpolated render path to easymocap format
    cameras = dotdict()
    for i in range(n_render_views):
        cam = dotdict()
        cam.K, cam.H, cam.W = Ks[i, 0], Hs[i, 0], Ws[i, 0]
        cam.R, cam.T = Rs[i, 0], Ts[i, 0]
        cam.t = ts[i, 0] if len(ts.shape) > 1 else ts[i]
        cam.n, cam.f = ns[i, 0], fs[i, 0]
        cam.D = Ds[i, 0] if Ds is not None else np.zeros((5, 1))
        cameras[f'{i:0{cam_digit}d}'] = to_numpy(cam)

    # Return the easymocap format cameras
    return cameras


def load_bodymodel(data_root: str, bodymodel_file: str, device='cpu'):
    # Import easymocap here since it's not a hard dependency, only if you want to use SMPL as prior
    from easymocap.config.baseconfig import load_object, Config

    if False:  # easymocap has been updated
        # Load config and convert the relative paths to absolute paths
        cfg_model = Config.load(join(data_root, bodymodel_file))  # whatever for now
        cfg_model.module = cfg_model.module.replace('SMPLHModelEmbedding', 'SMPLHModel')
        cfg_model.module = cfg_model.module.replace('SMPLLayerEmbedding', 'SMPLModel')

        # Cannot use relative path since easymocap is somewhere else and can be different on different machines
        easymocap_path = pkg_resources.get_distribution("easymocap").location
        for key, value in cfg_model.args.items(): cfg_model.args[key] = join(easymocap_path, value) if 'path' in key else value

        # Set device to cpu
        cfg_model.args.device = 'cpu'

        # Load actual body model
        bodymodel = load_object(cfg_model.module, cfg_model.args)

    if bodymodel_file:
        cfg_exp = Config.load(join(data_root, bodymodel_file))
    else:
        cfg_exp = Config.load(data_root)

    cfg_model = cfg_exp.args.at_final.load_body_model
    easymocap_path = pkg_resources.get_distribution("easymocap").location
    cfg_model.args.model_path = join(easymocap_path, cfg_model.args.model_path)
    cfg_model.args.regressor_path = join(easymocap_path, cfg_model.args.regressor_path)
    cfg_model.args.device = device

    body_loader = load_object(cfg_model.module, cfg_model.args)
    bodymodel = body_loader.smplmodel

    return bodymodel


def load_smpl(x):
    from easymocap.mytools.file_utils import read_annot
    from easyvolcap.utils.data_utils import to_tensor

    data = read_annot(x)[0]  # NOTE: singe person only
    for k, v in data.items():
        if isinstance(v, list):
            data[k] = np.asarray(v, dtype=np.float32)
    return to_tensor(dotdict(data))
