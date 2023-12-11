import argparse
from glob import glob
from os.path import join

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.engine import Output
from easyvolcap.utils.console_utils import log, run, stacktrace
# fmt: on

ratio = 0.5  # 1024 * 0.5 -> 512 x 512 or 1920 * 0.5 -> 960 x 540
light = [
    "main",
    "gym_entrance", "peppermint_powerplant_blue", "shanghai_bund", "courtyard_night", "large_corridor", "monks_forest", "pink_sunrise", "small_cave", "peppermint_powerplant", "st_peters_square_night", "studio_small", "veranda", "reinforced_concrete",
    'olat0004-0012', 'olat0004-0013', 'olat0004-0015', 'olat0004-0017', 'olat0004-0019', 'olat0004-0021', 'olat0004-0023', 'olat0004-0025', 'olat0004-0027',
    'olat0002-0012', 'olat0002-0013', 'olat0002-0015', 'olat0002-0017', 'olat0002-0019', 'olat0002-0021', 'olat0002-0023', 'olat0002-0025', 'olat0002-0027',
]
datasets = ['synthetic_human', 'mobile_stage']  # if empty, will render all datasets
# if empty, will render all characters
humans = \
    ['jody', 'josh', 'malcolm', 'megan', 'nathan'] +\
    ['xuzhen', 'black', 'white', 'purple']  # no time to train these
# ['xuzhen', 'black', 'white', 'purple', 'coat']

experiments = ['base', 'nerf', 'neuralbody', 'brute']  # if empty, will render all experiments


def main():
    global experiments
    global datasets
    global humans
    global light
    global ratio
    # global vis_args
    # global extra_args
    # global motion_args

    command = 'python run.py -t visualize -c {config} relighting True vis_novel_light True test_dataset_module lib.datasets.pose_dataset test_light "{light}" ratio {ratio} {vis_args} {extra_args} {motion_args}'
    vis_args = 'store_video_output False extra_prefix "eval" vis_alpha_map True vis_rendering_map True vis_albedo_map True vis_roughness_map True vis_normal_map True vis_shading_map True vis_envmap_map True vis_ext .exr print_render_progress True'
    motion_args = 'fix_material -1'
    # motion_args = 'num_eval_frame 300 frame_interval 2 test.frame_sampler_interval 50 begin_ith_frame 300 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz'
    # motion_args = 'num_eval_frame 300 frame_interval 3 test.frame_sampler_interval 50 begin_ith_frame 100 test_motion b11_wtl135.npz'
    extra_args = ''  # do not fix material when rendering with near training frames
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=experiments)
    parser.add_argument('--datasets', nargs='+', default=datasets)
    parser.add_argument('--humans', nargs='+', default=humans)
    parser.add_argument('--light', nargs='+', default=light)
    parser.add_argument('--ratio', type=float, default=ratio)
    parser.add_argument('--vis_args', type=str, default=vis_args)
    parser.add_argument('--extra_args', type=str, default=extra_args)
    parser.add_argument('--motion_args', type=str, default=motion_args)
    args = parser.parse_args()
    experiments = args.experiments
    datasets = args.datasets
    humans = args.humans
    light = args.light
    ratio = args.ratio
    vis_args = args.vis_args
    extra_args = args.extra_args
    motion_args = args.motion_args

    def has_any(string, stuff: list):
        if not stuff:
            return True
        for item in stuff:
            if item in string:
                return True
        return False

    configs = sorted(glob(join('configs', '**', "*")))
    configs = [c for c in configs if has_any(c, datasets) and has_any(c, experiments) and has_any(c, humans)]

    run('python scripts/tools/prepare_config.py')
    for c in configs:
        try:
            run(command.format(config=c, light=light, ratio=ratio, vis_args=vis_args, extra_args=extra_args, motion_args=motion_args))
        except:
            stacktrace()


if __name__ == '__main__':
    main()
