import torch
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.net_utils import typed, setup_deterministic
from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera, render_fdgs, render_diff_gauss
from easyvolcap.utils.console_utils import *


def test_illegal_memory_diag():
    repeat = 100
    N = 10000
    cov = torch.rand(N, 6, device='cuda') - 0.5  # trying to reproduce this
    occ = torch.rand(N, 1, device='cuda')

    K = torch.as_tensor([[736.5288696289062, 0.0, 682.7473754882812], [0.0, 736.4380493164062, 511.99737548828125], [0.0, 0.0, 1.0]], dtype=torch.float, device='cuda')
    R = torch.as_tensor([[0.9938720464706421, 0.0, -0.11053764075040817], [-0.0008741595083847642, 0.9999688267707825, -0.007859790697693825], [0.1105341762304306, 0.007908252067863941, 0.9938408732414246]], dtype=torch.float, device='cuda')
    T = torch.as_tensor([[-0.2975313067436218], [-1.2581647634506226], [0.2818146347999573]], dtype=torch.float, device='cuda')
    n = torch.as_tensor(0.02, dtype=torch.float, device='cuda')
    f = torch.as_tensor(1000.0, dtype=torch.float, device='cuda')
    W = torch.as_tensor(1366, dtype=torch.long, device='cuda')
    H = torch.as_tensor(768, dtype=torch.long, device='cuda')
    camera = convert_to_gaussian_camera(K, R, T, H, W, n, f,
                                        K.cpu(), R.cpu(), T.cpu(), H.cpu(), W.cpu(), n.cpu(), f.cpu()
                                        )

    for i in tqdm(range(repeat)):
        xyz = torch.rand(N, 3, device='cuda')
        rgb = torch.rand(N, 3, device='cuda')
        with open('fwd.dump', 'wb') as f:
            torch.save(dotdict(xyz3=xyz, rgb3=rgb, cov6=cov, occ1=occ, camera=camera), f)
            os.fsync(f)

        # render_fdgs(xyz, rgb, cov, occ, camera)  # will throw illegal memory access error
        render_diff_gauss(xyz, rgb, cov, occ, camera)  # will throw illegal memory access error


if __name__ == '__main__':
    my_tests(globals())
