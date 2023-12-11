from easyvolcap.utils.console_utils import *
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.net_utils import fair_patch_sample

import torch


def test_no_sampling():
    x, y = fair_patch_sample(10, 10, 5, 5)
    assert x == 0 and y == 0


def test_line_sampling(n_samples=10000):
    line = torch.zeros(50)
    for i in tqdm(range(n_samples)):
        x, y = fair_patch_sample(10, 10, 5, 50)
        line[x:x + 10] += 1
    print(line)
    print(line.std())


if __name__ == '__main__':
    my_tests(globals())
