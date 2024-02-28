# This file will convert the MVSNet depth estimation directory structure to easyvolcap's structure
# Will copy depth files to their corresponding places for easier depth fusion
# MVSNet will store depth files inside a single folder
# We will arrange them inside each of the camera folders

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_depth, save_image


@catch_throw
def main():
    args = dotdict()
    args.mvsnet_dir = '../cvp-mvsnet/outputs_pretrained/0008_01/depth_est'
    args.volcap_dir = 'data/renbody/0008_01/depths'
    args.image = '000000.jpg'
    args.convert = True
    args.convert_ext = '.exr'
    args = dotdict(vars(build_parser(args).parse_args()))


    files = [f for f in os.listdir(args.mvsnet_dir) if f.endswith('.pfm')]
    for f in tqdm(files):
        src = join(args.mvsnet_dir, f)
        dst = join(args.volcap_dir, f.split('.')[0], args.image.replace('.jpg', '.pfm'))
        os.makedirs(dirname(dst), exist_ok=True)

        if not args.convert:
            os.system(f'cp {src} {dst}')
        else:
            dpt = load_depth(src)
            save_image(dst.split('.')[0] + args.convert_ext, dpt)


if __name__ == '__main__':
    main()
