from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import generate_video

folder = '/mnt/data/home/xuzhen/projects/4k4d/4k4d.github.io.assets/compare_videos'
output = '/mnt/data/home/xuzhen/projects/4k4d/4k4d.github.io.assets/reencode_videos'

# cmd = 'ffmpeg -itsscale 0.5 -i {input} -c:v copy -y {output}'
cmd = 'ffmpeg -i {input} -c:v libx264 -crf 17 -pix_fmt yuv420p -y {output}'

from glob import glob

videos = glob(join(folder, '**/*.mp4'), recursive=True)

for vid in videos:
    target = vid.replace(folder, output)
    os.makedirs(dirname(target), exist_ok=True)
    run(cmd.format(input=vid, output=target))
    # print(target)