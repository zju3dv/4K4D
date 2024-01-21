from easyvolcap.utils.console_utils import *

@catch_throw
def main():
    args = dotdict()
    args.nerf_root = '/mnt/data/home/shuaiqing/Code/MultiNBFast/cache/Hospital/4'
    args.volcap_root = '/mnt/data/home/shuaiqing/Code/MultiNBFast/cache/Hospital/4'
    args.images_dir = 'images_train/00'
    args.transforms_file = 'transforms_train.json'
    args = dotdict(vars(build_parser(args).parse_args()))
    transforms = dotdict(json.load(open(join(args.nerf_root, args.transforms_file))))

    os.makedirs(join(args.volcap_root, args.images_dir), exist_ok=True)
    for i in tqdm(range(len(transforms.frames))):
        src = join(args.nerf_root, transforms.frames[i].file_path)
        tar = join(args.volcap_root, args.images_dir, f'{i:06d}.' + src.split('.')[-1])
        if exists(tar): os.remove(tar)
        os.symlink(relpath(src, dirname(tar)), tar)


if __name__ == '__main__':
    main()
