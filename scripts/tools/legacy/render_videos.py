# Render videos for demo
# We want novel views (rotating), with shade and normal map around
# Do not render attach ground, but render ground shadows
# We want results for multiple lightings, including some olats
# jody, josh, malcolm & xuzhen, black, white should be enough
# We need to render training & test frames & a dance sequence
# Render more videos, instead of longer videos?
# Ground shading tends to be quite slow, should render these carefully
# 6 characters * 3 sequences * 100 frames each -> ours -> 1800 frames * 8s per -> 4 hours??? -> 8 cards
# Maybe only render some of them with ground, one char, like jody? When performing ablation?
# Then, based on the rendering results, find comparison videos

import argparse
from glob import glob
from os.path import join

# fmt: off
import sys
sys.path.insert(0, '.')
from easyvolcap.engine import Output
from easyvolcap.utils.console_utils import log, run, stacktrace
from scripts.tools.render_relighting import ratio, light, humans, datasets, experiments
# fmt: on


# ratio = 0.5  # 1024 * 0.5 -> 512 x 512 or 1920 * 0.5 -> 960 x 540
# light = ["main", "gym_entrance", "peppermint_powerplant_blue", "shanghai_bund",
#          'olat0004-0012', 'olat0004-0013' 'olat0004-0015', 'olat0004-0017', 'olat0004-0019', 'olat0004-0021', 'olat0004-0023', 'olat0004-0025', 'olat0004-0027',
#          'olat0002-0012', 'olat0002-0013' 'olat0002-0015', 'olat0002-0017', 'olat0002-0019', 'olat0002-0021', 'olat0002-0023', 'olat0002-0025', 'olat0002-0027',
#          ]

# datasets = ['synthetic_human', 'mobile_stage']  # if empty, will render all datasets
# # if empty, will render all characters
# humans = \
#     ['jody', 'josh', 'malcolm', 'megan', 'nathan'] +\
#     ['xuzhen', 'black', 'white', 'purple', 'coat']
# # ['xuzhen', 'black', 'white', 'purple', 'coat']
# experiments = ['base', 'nerf', 'neuralbody', 'brute']  # if empty, will render all experiments

def main():
    global experiments
    global datasets
    global humans
    global light
    global ratio
    # global vis_args
    # global extra_args
    # global motion_args

    # 静态人体，动态光照，转光照 -> 贴背景，渲染地面 -> fast
    # 静态人体，静态的novel illumination，转view -> 贴背景，渲染地面-> slow
    # 动态人体，静态的original illumination -> 贴背景，渲染地面 -> slow
    # 动态人体，静态的novel illumination -> 贴背景，渲染地面 -> slow

    command = 'python run.py -t visualize -c {config} relighting True vis_novel_light True test_light "{light}" vis_novel_view True perform True ratio {ratio} {vis_args} {motion_args} {extra_args}'

    vis_args = 'store_video_output True extra_prefix "demo" vis_ground_shading True print_render_progress True vis_alpha_map True vis_rendering_map True vis_shading_map True vis_albedo_map True vis_normal_map True tonemapping_albedo True tonemapping_rendering True'
    motion_args = 'num_render_view 100 num_eval_frame 100 frame_interval 2 begin_ith_frame 300 test_motion gPO_sBM_cAll_d10_mPO0_ch01.npz'
    # motion_args = 'num_render_view 100 num_eval_frame 100 frame_interval 1'
    # motion_args = 'num_render_view 100 num_eval_frame 100 frame_interval 3 begin_ith_frame 100 test_motion b11_wtl135.npz'
    extra_args = ''

    # TODO: duplicated code
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
