"""
Convert fdgs npz to standard 4d gaussian ply
"""
import numpy as np
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.fdgs_utils import save_ply


@catch_throw
def main():
    args = dotdict(
        input='data/trained_model/fdgs_flame_salmon/latest.npz',
        output='data/result/fdgs_flame_salmon/POINT/latest.ply',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    npz = np.load(args.input)
    save_ply(npz, args.output)
    log(yellow(f'Saved converted PLY to: {blue(args.output)}'))


if __name__ == '__main__':
    main()
