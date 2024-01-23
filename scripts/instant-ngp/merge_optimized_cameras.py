import os
import cv2
import torch  # do we have a closed form solution for this?
import msgpack
import argparse
import numpy as np
import ujson as json
import torch.nn.functional as F

from tqdm import tqdm
from os.path import join
from torch.optim import Adam
from functools import partial
from multiprocessing.pool import ThreadPool
from smplx.lbs import batch_rodrigues, batch_rigid_transform
from dwb.utilities.utils import resize_images_tensor, get_mapping_func, apply_mapping_func


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
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
            self._write('  data: [{}]'.format(', '.join(['{:.3f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))

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
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format(cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Tvec = extri.read('T_{}'.format(cam))

        Rvec = extri.read('R_{}'.format(cam))
        if Rvec is not None:
            R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read('Rot_{}'.format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['Rvec'] = Rvec
        cams[cam]['T'] = Tvec
        cams[cam]['center'] = - Rvec.T @ Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams


def write_camera(camera, path):
    from os.path import join
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/nas/datasets/mgtv_nvs/mocap_mgtv_val_gt', help="data root")  # assume the outer level is not frames
parser.add_argument('--ngp_dir', default='ngp', nargs='+', help="where does all these ngp params live")  # assume the outer level is not frames
parser.add_argument('--sequences', default=['F1_06', "F2_07"], nargs='+', help="which sequences to merge")  # assume the outer level is not frames
parser.add_argument('--transforms_file', default='transforms.json', help="optimized conversion parameters")  # TODO: maybe also export relevant params?
parser.add_argument('--snapshot_file', default='vdfq_6200.msgpack', help="the optimized snapshot model file name")  # TODO: maybe also export relevant params?
parser.add_argument('--extrinsics_file', default='camera_params/extrinsics.json', help="optimized conversion parameters")
parser.add_argument('--extri_optimized_output', default='extri_optimized.json', help="merged transformation in easymocap format")  # assume an outer level of frames
parser.add_argument('--output_dir', default=['/nas/datasets/mgtv_nvs/mocap_mgtv_val', '/nas/datasets/mgtv_nvs/mocap_mgtv_test_a', '/nas/datasets/mgtv_nvs/mocap_mgtv_test_b'], nargs='+', help="where to output the merged extri_optimized.yml")  # assume an outer level of frames


parser.add_argument('--save_inverse_images', action='store_true')

parser.add_argument('--no_merge_extrinsics', action='store_false', dest="merge_extrinsics")
parser.add_argument('--no_merge_linear_map', action='store_false', dest="merge_linear_map")
parser.add_argument('--no_increamental_save', action='store_false', dest="increamental_save")

parser.add_argument('--awb_resize', default=-1, type=int)
parser.add_argument('--n_cameras', default=92, type=int)
parser.add_argument('--kernel', default=11, type=int)
parser.add_argument('--background_dir', default='/nas/datasets/mgtv_nvs/background')  # output to this guy here

grp_bkgd_map = {
    'F1_F2': ['F1', 'F2'],  # MARK: only F series, skip optim for M series
    'M1_M2_M3': ['M1', 'M2', 'M3']
}

parser.add_argument('--device', default='cuda', help="where to perform the optimization")
parser.add_argument('--n_steps', default=200, type=int, help="how many optimization steps to take")
parser.add_argument('--lr', default=1e-2, type=float, help="learning rate of the optimizer")
args = parser.parse_args()


def load_image(img_path: str, denoise=False):
    if denoise:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img[..., :3] = cv2.fastNlMeansDenoisingColored(img[..., :3], None, 10, 10, 7, 21)
        img = (img[..., [2, 1, 0]] / 255).astype(np.float32)  # BGR to RGB
    else:
        img = (cv2.imread(img_path, cv2.IMREAD_COLOR)[..., [2, 1, 0]] / 255).astype(np.float32)  # BGR to RGB
    return img


def load_mask(img_path: str): return (cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[..., None] > 128).astype(bool)  # BGR to RGB


def load_mask_from_image(img_path: str): return (cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[..., -1:] > 128).astype(bool)  # BGR to RGB


def save_image(img_path: str, img: np.ndarray):
    if img.ndim == 4:
        assert img.shape[0] == 1, f'Unsupported shape: {img.shape}, batch size should at most be 1'
        img = img[0]
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    return cv2.imwrite(img_path, img.clip(0, 1)[..., [2, 1, 0]] * 255, [cv2.IMWRITE_JPEG_QUALITY, 100])


def save_mask(img_path: str, msk: np.ndarray):
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    return cv2.imwrite(img_path, msk[..., 0][..., 0] * 255, [cv2.IMWRITE_JPEG_QUALITY, 100])


def list_to_tensor(x: list, device='cuda'): return torch.tensor(np.stack(x), device=device).permute(0, 3, 1, 2)


def tensor_to_numpy(x: torch.Tensor): return x.permute(0, 2, 3, 1).detach().cpu().numpy()


def parallel_execution(*args, num_processes=64, action=lambda *args, **kwargs: None, print_progress=True, **kwargs):
    # NOTE: we expect first arg / or kwargs to be distributed
    # NOTE: print_progress arg is reserved

    results = []
    async_results = []
    pool = ThreadPool(processes=num_processes)

    valid_arg = args[0] if isinstance(args[0], list) else next(iter(kwargs.values()))  # TODO: search through them all

    # Spawn threads
    for i in range(len(valid_arg)):  # might be lambda x: np.split(x.permute(0, 2, 3, 1).detach().cpu().numpy(), len(x))
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == len(valid_arg) else arg) for arg in args]
        action_kwargs = {key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == len(valid_arg) else kwargs[key]) for key in kwargs}
        async_result = pool.apply_async(action, action_args, action_kwargs)
        async_results.append(async_result)

    # Join threads and get return values
    def maybe_tqdm(x): return tqdm(x) if print_progress else x
    for async_result in maybe_tqdm(async_results):
        results.append(async_result.get())  # will sync the corresponding thread
    pool.close()
    pool.join()
    return results  # might be lambda x: torch.tensor(np.stack(x), device='cuda').permute(0, 3, 1, 2)


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
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
            self._write('  data: [{}]'.format(', '.join(['{:.8f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))

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
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format(cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Tvec = extri.read('T_{}'.format(cam))

        Rvec = extri.read('R_{}'.format(cam))
        if Rvec is not None:
            R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read('Rot_{}'.format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['Rvec'] = Rvec
        cams[cam]['T'] = Tvec
        cams[cam]['center'] = - Rvec.T @ Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams


def write_camera(camera, path, intri_name='intri.yml', extri_name='extri.yml'):
    from os.path import join
    intri_name = join(path, intri_name)
    extri_name = join(path, extri_name)
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])


def qvec2rotmat(qvec: np.ndarray):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R: np.ndarray):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def rotmat(a: np.ndarray, b: np.ndarray):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def get_rigid_transforms(angle_axis, translation):
    R = batch_rodrigues(angle_axis)  # B, 3, 3
    T = translation[..., None]  # B, 3, 1
    RT = torch.cat([R, T], dim=-1)  # B, 3, 4
    pad = RT.new_zeros([R.shape[0], 1, 4], dtype=torch.float)
    pad[..., -1] = 1
    middle = torch.cat([RT, pad], dim=-2)
    return middle


if args.merge_extrinsics:
    # for every sequence, will load the pivot point: transforms_file and snapshot_file, save relevant variables
    # Close method of the file will be called when object is destroyed, as json.load expects only read method on input object.
    # https://stackoverflow.com/questions/7395542/is-explicitly-closing-files-important
    for seq in args.sequences:
        frame_dirs = sorted([join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.startswith(seq)])
        frame_dirs = [f for f in frame_dirs if os.path.isdir(f)]

        # prepare paths for the first frame of a sequence
        ngp_dir = join(frame_dirs[0], args.ngp_dir)
        snapshot = msgpack.load(open(join(ngp_dir, args.snapshot_file), 'rb'))
        transforms = json.load(open(join(ngp_dir, args.transforms_file)))
        target = np.array(json.load(open(join(ngp_dir, args.extrinsics_file)))).astype(np.float32)  # N, 4, 4

        # load important variables for this sequence
        # scale = snapshot['snapshot']['nerf']['dataset']['scale']
        # offset = np.array(snapshot['snapshot']['nerf']['dataset']['offset'])  # 3,
        ori_up = np.array(transforms['ori_up'])
        totp = np.array(transforms['totp'])
        avglen = np.array(transforms['avglen'])

        # optimize a middle matrix so that all frames in a sequence has an aligned camera space
        # prepare source transforms
        # prepare target transforms
        n_middles = len(frame_dirs) - 1
        source = np.array([json.load(open(join(frame, args.ngp_dir, args.extrinsics_file))) for frame in frame_dirs[1:]]).astype(np.float32)  # B, N, 3, 4

        # prepare target for optimization
        angle_axis = torch.zeros([n_middles, 3], dtype=torch.float32, device=args.device, requires_grad=True)
        translation = torch.zeros([n_middles, 3], dtype=torch.float32, device=args.device, requires_grad=True)
        target = torch.tensor(target, device=args.device)
        source = torch.tensor(source, device=args.device)

        # prepare optimizer
        optim = Adam([angle_axis, translation], args.lr)

        # perform optimization
        p = tqdm(range(args.n_steps))
        for i in p:
            middle = get_rigid_transforms(angle_axis, translation)
            prediction = middle[:, None] @ source  # B, N, 4, 4
            loss = (prediction - target[None]).pow(2).sum(dim=-1).sum(dim=-1).mean()  # F-norm

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            p.set_description('loss: {:.8f}'.format(loss.item()))

        with torch.no_grad():
            middle = get_rigid_transforms(angle_axis, translation)
            prediction = middle[:, None] @ source  # B, N, 4, 4
        prediction = prediction.detach().cpu().numpy()  # B, N, 4, 4

        # getting merged extrinsics
        # TODO: batch this
        merged = np.array([[np.concatenate([rotmat2qvec(cam[:3, :3]), cam[:3, 3]], axis=0) for cam in frame] for frame in prediction])  # B, N, 7
        merged = merged.mean(axis=0)  # get mean on quaternion, N, 7
        merged = np.array([np.concatenate([qvec2rotmat(cam[:4]), cam[4:][..., None]], axis=1) for cam in merged])  # N, 3, 4

        # # TODO: export merged.json at corresponding sequence
        # json.dump(merged.tolist(), open(join(args.data_dir, f'{seq}_merged.json'), 'w'), indent=4)

        c2w = merged
        pad = np.zeros([c2w.shape[0], 1, 4], dtype=np.float32)
        pad[..., -1] = 1
        c2w = np.concatenate([c2w, pad], -2)  # 4, 4 # restore to transforms.json format
        c2w[:, :3, 3] /= 4. / avglen
        c2w[:, :3, 3] += totp

        R = rotmat(ori_up, [0, 0, 1])  # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1
        # TODO: batch this
        c2w = np.array([np.matmul(R.T, c2w[i]) for i in range(len(c2w))])  # restore to c2w format

        c2w[:, 2, :] *= -1  # flip whole world upside down
        c2w = c2w[:, [1, 0, 2, 3], :]  # swap back y and z:w
        c2w[:, :3, 2] *= -1  # flip the y and z axis
        c2w[:, :3, 1] *= -1

        # TODO: batch this
        w2c = np.array([np.linalg.inv(c2w[i]) for i in range(len(c2w))])  # 4, 4
        R, T = w2c[:, :3, :3], w2c[:, :3, 3:]  # restore to easymocap format, N

        # write to extri_optimized.yml
        for output in args.output_dir:
            output_frames = sorted([f for f in os.listdir(output) if f.startswith(seq)])
            output_frames = [f for f in output_frames if os.path.isdir(join(output, f))]
            for output_frame in output_frames:
                output_frame_dir = join(output, output_frame)

                print(output_frame_dir)
                cams = read_camera(join(output_frame_dir, 'intri.yml'), join(output_frame_dir, 'extri.yml'))
                basenames = cams['basenames']
                del cams['basenames']
                for cam_id in range(len(basenames)):
                    cam_name = f"{cam_id+1:02d}"
                    assert cam_name == basenames[cam_id]
                    print(f'{cam_name}')
                    print(f'R:\n{R[cam_id]}')
                    print(f'original R:\n{cams[cam_name]["R"]}')
                    print(f'T:\n{T[cam_id]}')
                    print(f'original T:\n{cams[cam_name]["T"]}')
                    cams[cam_name]['R'] = R[cam_id]
                    cams[cam_name]['T'] = T[cam_id]

                print(output_frame_dir)
                write_camera(cams, output_frame_dir, 'intri.yml', 'extri_optimized.yml')

if args.merge_linear_map:

    # for every sequence, will load the pivot point: transforms_file and snapshot_file, save relevant variables
    # Close method of the file will be called when object is destroyed, as json.load expects only read method on input object.
    # https://stackoverflow.com/questions/7395542/is-explicitly-closing-files-important
    for grp in grp_bkgd_map:
        print(f'processing group: {grp}')

        ag_dir = join(args.background_dir, grp)

        # if args.increamental_save and os.path.exists(join(ag_dir, 'm_awb_dict.npy')):
        #     m_awb_dict = np.load(join(ag_dir, 'm_awb_dict.npy'), allow_pickle=True).item()
        #     m_awb_inv_dict = np.load(join(ag_dir, 'm_awb_inv_dict.npy'), allow_pickle=True).item()
        # else:
        m_awb_dict = {}
        m_awb_inv_dict = {}

        # iterate through all cameras
        for cam_id in range(args.n_cameras):
            cam = f"{cam_id+1:02d}"
            print(f'processing camera: {cam}')

            # prepare action group for this sequence
            sources_list = []
            targets_list = []

            # prepare all source images and target images
            for seq in grp_bkgd_map[grp]:
                # prepare all frames
                frame_dirs = sorted([join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.startswith(seq)])
                frame_dirs = [f for f in frame_dirs if os.path.isdir(f)]

                # camera file list
                for frame_dir in frame_dirs:
                    sources_list.append(join(frame_dir, args.ngp_dir, f'{cam}.png'))
                    targets_list.append(join(frame_dir, args.ngp_dir, 'eval', f'{cam}.png'))

            sources = list_to_tensor(parallel_execution(sources_list, action=partial(load_image, denoise=False)), args.device)
            targets = list_to_tensor(parallel_execution(targets_list, action=partial(load_image, denoise=False)), args.device)
            mask = list_to_tensor(parallel_execution(sources_list, action=load_mask_from_image), args.device)
            mask = ~F.conv2d((~mask).float(), torch.ones(1, 1, args.kernel, args.kernel, dtype=torch.float, device=args.device), padding=args.kernel // 2).bool()  # dilate the mask with a 21x21 kernel

            if args.awb_resize > 0:
                sources = resize_images_tensor(sources, s=args.awb_resize, mode='bilinear')
                targets = resize_images_tensor(targets, s=args.awb_resize, mode="bilinear")
                mask = resize_images_tensor(mask.float(), s=args.awb_resize, mode='nearest').bool()

            mask = mask.permute(0, 2, 3, 1)[..., 0]
            sources = sources.permute(0, 2, 3, 1)[mask]
            targets = targets.permute(0, 2, 3, 1)[mask]

            alpha, beta = None, None

            # # compute exposure correction on foreground pixels
            # unique, counts = (sources * 255).clip(0, 255).to(torch.uint8).unique(sorted=True, return_counts=True)
            # percentage = counts.cumsum(dim=0) / sources.numel()
            # source_high = unique[(percentage < 0.90).sum()] / 255  # for any range
            # source_low = unique[(percentage < 0.10).sum()] / 255  # for 0-1
            # unique, counts = (targets * 255).clip(0, 255).to(torch.uint8).unique(sorted=True, return_counts=True)
            # percentage = counts.cumsum(dim=0) / targets.numel()
            # target_high = unique[(percentage < 0.90).sum()] / 255  # for any range
            # target_low = unique[(percentage < 0.10).sum()] / 255  # for 0-1
            # alpha = (target_high - target_low) / (source_high - source_low)
            # beta = -source_low * alpha + target_low
            # sources = alpha * sources + beta

            appended = torch.cat([sources.view(-1, 1), torch.ones_like(sources.view(-1, 1))], dim=-1)  # N, 2
            reshaped = targets.view(-1, 1)  # N, 1
            alpha_beta = torch.inverse(appended.T @ appended) @ appended.T @ reshaped  # 2, 1
            alpha, beta = alpha_beta
            sources = alpha * sources + beta

            sources = sources.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            alpha, beta = alpha.item(), beta.item()
            print(f'alpha: {alpha}, beta: {beta}')

            # targets = alpha * sources + beta
            # targets = sources @ [alpha, beta].T
            # [alpha, beta] = np.linalg.inv(sources.T @ sources) @ sources.T @ targets

            print(f'solving forward linear system mapping: {sources.size // 3}, 11 -> {targets.size // 3}, 3')
            m_awb = get_mapping_func(sources, targets, device=args.device)
            mapped = apply_mapping_func(sources, m_awb, device=args.device)
            psnr = 10 * -np.log(((targets - mapped) ** 2).mean()) / np.log(10)
            print(f'psnr of mapped and target: {psnr}')

            print(f'solving backward linear system mapping: {targets.size // 3}, 11 -> {sources.size // 3}, 3')
            m_awb_inv = get_mapping_func(mapped, sources, device=args.device)
            mapped = apply_mapping_func(mapped, m_awb_inv, device=args.device)
            psnr = 10 * -np.log(((sources - mapped) ** 2).mean()) / np.log(10)
            print(f'psnr of mapped and target: {psnr}')

            if alpha is not None and beta is not None:
                m_awb_dict[cam] = (m_awb, alpha, beta)  # NOTE: tuple or list is used to differentiate between pipeline and multiple arguments
                m_awb_inv_dict[cam] = (m_awb_inv, alpha, beta)
            else:
                m_awb_dict[cam] = m_awb
                m_awb_inv_dict[cam] = m_awb_inv

            if args.save_inverse_images:
                images = parallel_execution(sources_list, action=partial(load_image, denoise=False))
                mapped = parallel_execution(images, action=lambda x: apply_mapping_func(alpha * x + beta, m_awb))
                inverse = parallel_execution(mapped, action=lambda x: (apply_mapping_func(x, m_awb_inv) - beta) / alpha)

                parallel_execution([t.replace('eval', 'inv') for t in targets_list], inverse, action=save_image)

        if args.increamental_save:
            m_awb_dict_ori = np.load(join(ag_dir, 'm_awb_dict.npy'), allow_pickle=True).item()
            m_awb_inv_dict_ori = np.load(join(ag_dir, 'm_awb_inv_dict.npy'), allow_pickle=True).item()
            for cam in m_awb_dict_ori:
                # NOTE: tuple or list is used to differentiate between pipeline and multiple arguments
                m_awb_dict[cam] = [m_awb_dict_ori[cam], m_awb_dict[cam]]  # will automatically chain these (boosting dimensionality in between)
                m_awb_inv_dict[cam] = [m_awb_inv_dict[cam], m_awb_inv_dict_ori[cam]]

        np.save(join(ag_dir, 'm_awb_dict.npy'), m_awb_dict, allow_pickle=True)
        np.save(join(ag_dir, 'm_awb_inv_dict.npy'), m_awb_inv_dict, allow_pickle=True)
