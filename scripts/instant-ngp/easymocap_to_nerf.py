# 将单帧数据转换为Nerf格式
from easymocap.mytools.camera_utils import read_cameras, Undistort, read_camera
from easymocap.mytools.file_utils import save_json
from os.path import join
import numpy as np
import cv2
import os
from tqdm import tqdm
import math
from glob import glob


def convert_K(K):
    fl_x = K[0, 0]
    fl_y = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    AABB_SCALE = 1

    w, h = 1024, 1024
    angle_x = math.atan(w/(fl_x*2))*2
    angle_y = math.atan(h/(fl_y*2))*2
    fovx = angle_x*180/math.pi
    fovy = angle_y*180/math.pi
    k1, k2, p1, p2 = 0., 0., 0., 0.
    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": AABB_SCALE,
    }
    return out


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def convertRT(RT0):
    RT = np.eye(4)
    RT[:3] = RT0
    c2w = np.linalg.inv(RT)
    c2w[0:3, 1] *= -1
    c2w[0:3, 2] *= -1  # flip the y and z axis
    # c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
    # c2w[2, :] *= -1  # flip whole world upside down
    return c2w


# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db):
    da = da/np.linalg.norm(da)
    db = db/np.linalg.norm(db)
    c = np.cross(da, db)
    denom = (np.linalg.norm(c)**2)
    t = ob-oa
    ta = np.linalg.det([t, db, c])/(denom+1e-10)
    tb = np.linalg.det([t, da, c])/(denom+1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db)*0.5, denom


def normalize_cameras(out):
    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0
    totp = [0, 0, 0]
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p*w
                totw += w
    totp /= totw
    print(totp)  # the cameras are looking at totp
    # totp = np.array([-1.0, -0.1, -0.1])
    print(totp)  # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp
    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= len(out['frames'])
    avglen = 1
    print("avg camera distance from origin ", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 1./avglen     # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()


if __name__ == '__main__':
    if False:
        data = '/nas/datasets/multi-neuralbody/female-jump'
        out = 'nerf_synthetic/ballet'
        nf = 0
    elif False:
        data = '/nas/datasets/multi-neuralbody/handstand'
        out = 'nerf_synthetic/handstand'
        nf = 0
    elif False:
        data = '/nas/datasets/EasyMocap/static1p'
        out = '/nas/datasets/EasyMocap/static1p-nerf'
        nf = 0
        mask = 'mask'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--nf', type=int, default=0)
    parser.add_argument('--mask', type=str, default='mask')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    data = args.path
    out = args.out
    nf, mask = args.nf, args.mask

    with_mask = True
    if with_mask:
        out_ext = '.png'
        aabb_scale = 1
    else:
        out_ext = '.jpg'
        aabb_scale = 4
        out += '-back'
    os.makedirs(out, exist_ok=True)
    cameras = read_camera(join(data, 'intri.yml'), join(data, 'extri.yml'))

    subs = cameras['basenames']
    cameras = {key: cameras[key] for key in subs}
    for split in ['train']:
        annots = {
            'aabb_scale': aabb_scale,
            'frames': []}
        K = cameras[subs[0]]['K']
        annots.update(convert_K(K))

        for sub in tqdm(subs, desc=split):
            camera = cameras[sub]
            filename = '{}/{}{}'.format(split, sub, out_ext)
            imgname = join(data, sub, '{:06d}.jpg'.format(nf))
            assert os.path.exists(imgname), imgname
            K, dist = camera['K'], camera['dist']
            img = cv2.imread(imgname)
            b = sharpness(img)
            if with_mask:
                msknames = glob(join(data, mask, sub, '{:06d}*'.format(nf)))
                msks = [cv2.imread(mskname, 0) for mskname in msknames]
                msk = np.zeros_like(img[:, :, 0])
                for m in msks:
                    m[m > 0] = 255
                    msk = msk | m
                img = np.dstack([img, msk[..., None]])
            img = Undistort.image(img, K, dist)
            outname = join(out, split, sub+out_ext)
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            cv2.imwrite(outname, img)
            # convert RT
            c2w = convertRT(camera['RT'])
            info = {
                'file_path': filename,
                'sharpness': b,
                'transform_matrix': c2w
            }
            annots['frames'].append(info)
        normalize_cameras(annots)
        save_json(join(out, 'transforms_' + split+'.json'), annots)
