"""
Perform voxel reconstruction on input point cloud
"""

from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(

    )
    args = vars(build_parser(args, description=__doc__).parse_args())


if __name__ == '__main__':
    main()
