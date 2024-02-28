"""
Copy as list of files (folders), preserving directory structure
"""
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        files=[
            'easyvolcap/runners/evaluators/volumetric_video_evaluator.py',
            'easyvolcap/runners/optimizers.py',
            'easyvolcap/runners/schedulers.py',
            'easyvolcap/utils/loss_utils.py',
            'scripts/colmap/easymocap_to_colmap.py',
            'scripts/preprocess/neural3dv_to_easyvolcap.py',
            'scripts/tools/copy_list_of_files.py',
        ],
        source='.',
        target='../easyvolcap-zju3dv',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    for f in tqdm(args.files):
        src = join(args.source, f)
        tar = join(args.target, f)
        os.makedirs(dirname(tar), exist_ok=True)
        if isfile(src):
            shutil.copy(src, tar)
        else:
            shutil.copytree(src, tar)


if __name__ == '__main__':
    main()
