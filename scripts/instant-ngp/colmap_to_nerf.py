#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import cv2
import math
import json
import argparse
import numpy as np
from os.path import join

from easyvolcap.utils.colmap_utils import read_model, read_cameras_binary, write_cameras_text, read_images_binary, write_images_text
from easyvolcap.utils.console_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument('--data_root', default='data/NHR/sport_1_easymocap')
    parser.add_argument("--out_dir", default="ngp", help="output directory")
    parser.add_argument("--images", default="ngp", help="input path to the images")
    parser.add_argument("--text", default="ngp", help="input path to the colmap text files (set automatically if run_colmap is used)")

    parser.add_argument("--aabb_scale", default=1, choices=["1", "2", "4", "8", "16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
    parser.add_argument("--keep_colmap_coords", action="store_true", help="keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering)")
    parser.add_argument('--aabb', default='')

    args = parser.parse_args()
    args.out_dir = join(args.data_root, args.out_dir)
    args.images = join(args.data_root, args.images)
    args.text = join(args.data_root, args.text)
    return args


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


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
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa: np.ndarray, da: np.ndarray, ob: np.ndarray, db: np.ndarray):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def main(args, image_text='images.txt', camera_text='cameras.txt', out_json='transforms.json', up_in=None, compute_sharpness=True, totp=None, avglen=None):
    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)
    IMAGE_FOLDER = args.images
    TEXT_FOLDER = args.text

    OUT_PATH = os.path.join(args.out_dir, out_json)
    print(f"outputting to {OUT_PATH}...")
    OUT_DIR = os.path.dirname(OUT_PATH)
    os.system(f'mkdir -p {OUT_DIR}')
    print(f'creating dir: {OUT_DIR}')
    with open(os.path.join(TEXT_FOLDER, camera_text), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            line = line.strip()
            if line and line[0] == "#":
                continue
            if not line:
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV" or els[1] == "FULL_OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    aabb = None
    if os.path.exists(args.aabb):
        aabb = np.load(args.aabb)
        aabb[0] -= 0.05  # 5cm
        aabb[1] += 0.05  # 5cm
        aabb = aabb.tolist()
        print(f'aabb: {aabb}')

    with open(os.path.join(TEXT_FOLDER, image_text), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
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
            "frames": [],
        }

        if aabb is not None:
            out['aabb'] = aabb

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line and line[0] == "#":
                continue
            i = i + 1
            if not line:
                continue
            if i < SKIP_EARLY * 2:
                continue
            if i % 2 == 1:
                elems = line.split(" ")  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                # name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER, OUT_DIR)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)

                if not args.keep_colmap_coords:
                    c2w[0:3, 2] *= -1  # flip the y and z axis
                    c2w[0:3, 1] *= -1
                    c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
                    c2w[2, :] *= -1  # flip whole world upside down

                    up += c2w[0:3, 1]

                frame = {"file_path": name, "transform_matrix": c2w}

                if compute_sharpness:
                    b = sharpness(name)
                    print(elems[9], "sharpness=", b)
                    frame['sharpness'] = b

                out["frames"].append(frame)

    with open(os.path.join(TEXT_FOLDER, camera_text), "r") as f:
        i = 0
        for idx, line in enumerate(f):
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            line = line.strip()
            if line and line[0] == "#":
                continue
            i = i + 1
            if not line:
                continue
            if i < SKIP_EARLY * 2:
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if (els[1] == "SIMPLE_RADIAL"):
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif (els[1] == "RADIAL"):
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif (els[1] == "OPENCV" or (els[1] == "FULL_OPENCV")):
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])

            out["frames"][i - 1]['fl_x'], out["frames"][i - 1]['fl_y'], out["frames"][i - 1]['k1'], out["frames"][i - 1]['k2'], out["frames"][i - 1]['p1'], out["frames"][i - 1]['p2'], out["frames"][i - 1]['cx'], out["frames"][i - 1]['cy'] = \
                fl_x, fl_y, k1, k2, p1, p2, cx, cy

            out["frames"][i - 1]['w'], out["frames"][i - 1]['h'] = w, h

    nframes = len(out["frames"])

    if args.keep_colmap_coords:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)  # flip cameras (it just works)
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        if up_in is None:
            up = up / np.linalg.norm(up)
        else:
            up = up_in

        print("up vector was", up)
        R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
        R = np.pad(R, [0, 1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

        if totp is None:
            # find a central point they are all looking at
            print("computing center of attention...")
            totw = 0.0
            totp = np.array([0.0, 0.0, 0.0])
            # TODO: vectorize this to make it faster, this is not very tom94
            for f in out["frames"]:
                mf = f["transform_matrix"][0:3, :]
                for g in out["frames"]:
                    mg = g["transform_matrix"][0:3, :]
                    p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                    if w > 0.01:
                        totp += p * w
                        totw += w
            totp /= totw
        else:
            print('reusing center of attension...')
        print(totp)  # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp

        if avglen is None:
            avglen = 0.
            for f in out["frames"]:
                avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
            avglen /= nframes

        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"
        out['ori_up'] = up.tolist()
        out['totp'] = totp.tolist()
        out['avglen'] = avglen.tolist()

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    print(f"writing {OUT_PATH}")

    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

    return up, totp, avglen


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(os.path.join(args.text, 'images.bin')) and \
       not os.path.exists(os.path.join(args.text, 'images.txt')):
        # c, i, p = read_model(args.text) # assuming text and images are stored at the same place
        write_cameras_text(read_cameras_binary(args.text + '/' + 'cameras.bin'), args.text + '/' + 'cameras.txt')
        write_images_text(read_images_binary(args.text + '/' + 'images.bin'), args.text + '/' + 'images.txt')
    if os.path.exists(os.path.join(args.text, 'images.txt')):
        up, totp, avglen = main(args, 'images.txt', 'cameras.txt', 'transforms.json', compute_sharpness=False)
