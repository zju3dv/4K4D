"""
Convert shuaiqing pth file to standard gaussian ply
"""
import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.gaussian_utils import GaussianModel, save_gs


@catch_throw
def main():
    args = dotdict(
        input='/home/shuaiqing/Code/MultiNBFast/output/actorshq/debug/model_init_wotrain.pth',
        output='/home/shuaiqing/Code/MultiNBFast/output/actorshq/debug/model_init_wotrain.ply',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    shuai = torch.load(args.input, map_location='cpu')
    gs = dotdict()
    gs._xyz = shuai['gaussian.xyz'][..., 0, :]
    gs._scaling = shuai['gaussian.scaling']
    gs._rotation = shuai['gaussian.rotation'][..., 0, :]
    gs._opacity = shuai['gaussian.opacity']
    gs._features_dc = shuai['gaussian.colors'][..., None]
    gs._features_rest = shuai['gaussian.shs']
    save_gs(gs, args.output)

    log(yellow(f'Converted GS saved to {blue(args.output)}'))


if __name__ == '__main__':
    main()
