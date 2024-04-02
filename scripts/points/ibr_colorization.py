"""
Load the pretrained model of ENeRF's IBR module
Will serve as a blender for image based rendering of the colors
How do we determine which view to use for the blending?
Or just use median value?
This seems counter-intuitive
Assuming shared camera parameters for now

How to get the correct source views?
Assuming you've got a good position, we unproject all points on all views, then perform argsort on depth
For every point, then find the closest ranking view, and use that view's color
This way we can deal with non-uniform camera distribution
"""
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        data_root='data/mobile_stage',
        images_dir='images',
        input='surfs6k',
        output='surfs6k',
        n_srcs=4,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))


if __name__ == '__main__':
    main()
