import torch
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.net_utils import typed, setup_deterministic
from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera
from easyvolcap.utils.console_utils import *

# fmt: off
import sys
sys.path.append('.')
from tests.cuda_illegal_memory_access_tests import render_diff_gauss
# fmt: on


def test_illegal_memory_rendering():
    input = torch.load('fwd.dump')
    input = dotdict(input)
    xyz, rgb, cov, occ, camera = input.xyz3, input.rgb3, input.cov6, input.occ1, input.camera
    breakpoint()
    rgb, acc, dpt, meta = render_diff_gauss(xyz, rgb, cov, occ, camera)
    # rgb, acc, dpt, meta = render_diff_gauss(input.xyz3, input.rgb3, input.cov6, input.occ1, input.camera)

    # N = len(input.xyz3)
    # # input.xyz3.requires_grad_()
    # # input.rgb3.requires_grad_()
    # # input.cov6.requires_grad_()
    # # input.occ1.requires_grad_()
    # first_N_points = 1
    # while first_N_points < 2 * N:
    #     log(f'Rendering the first {min(N, first_N_points)} points out of {N}')
    #     # for i in tqdm(range(repeat)):
    #     rgb, acc, dpt, meta = render_diff_gauss(input.xyz3[:first_N_points], input.rgb3[:first_N_points], input.cov6[:first_N_points], input.occ1[:first_N_points], input.camera)
    #     first_N_points *= 2


if __name__ == '__main__':
    setup_deterministic()
    my_tests(globals())
