import argparse
import cv2
import os
from os.path import join
from glob import glob
from tqdm import tqdm


def extract_video(videoname, path, start, end, step, sub=[]):
    base = os.path.basename(videoname)[:2]  # only first two chars
    if sub:
        if base not in sub:
            print(f'>> skipping {base}')
            return base
    if not os.path.exists(videoname):
        return base
    outpath = join(path, 'images', base)
    if os.path.exists(outpath) and len(os.listdir(outpath)) > 0:
        num_images = len(os.listdir(outpath))
        print('>> exists {} frames'.format(num_images))
        return base
    else:
        os.makedirs(outpath, exist_ok=True)
    video = cv2.VideoCapture(videoname)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for cnt in tqdm(range(totalFrames), desc='{:10s}'.format(os.path.basename(videoname))):
        ret, frame = video.read()
        if cnt < start: continue  # TODO: so wasteful
        if cnt >= end: break  # TODO: so wasteful
        if (cnt - start) % step: continue  # mod returns 0 -> on step
        if not ret: continue
        cv2.imwrite(join(outpath, '{:06d}.jpg'.format(cnt - start)), frame[::-1, ::-1])  # rotate 180 degrees
    video.release()
    return base


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--end', default=10000, type=int)
parser.add_argument('--step', default=1, type=int)
parser.add_argument('--sub', default=[], nargs='*')
args = parser.parse_args()

videos = glob(join(args.input, 'videos', '*.mp4'))
for video in videos:
    extract_video(video, args.output, args.start, args.end, args.step, args.sub)
