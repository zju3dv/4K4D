from easyvolcap.utils.console_utils import *
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.data_utils import load_pts
from easyvolcap.utils.timer_utils import timer

import cupy
import torch
import cupy_knn

from pytorch3d.ops import knn_points

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

device = 'cuda'
pts = load_pts('assets/meshes/bunny.ply')[0]  # only the vertices matter
pts = torch.as_tensor(pts, device=device)[None]
K = 10
timer.disabled = False


def test_torch_cupy_interop():
    global pts
    pts_cupy = cupy.asarray(pts)
    print(pts_cupy)
    assert pts.__cuda_array_interface__['data'][0] == pts_cupy.__cuda_array_interface__['data'][0]

    pts = torch.as_tensor(pts_cupy, device=device)
    assert pts.__cuda_array_interface__['data'][0] == pts_cupy.__cuda_array_interface__['data'][0]


def test_cupy_knn_results():
    d2, idx, nn = knn_points(pts, pts, K=K, return_nn=False, return_sorted=True)

    pts_cupy = cupy.asarray(pts)
    lbvh = cupy_knn.LBVHIndex(leaf_size=32,
                              compact=False,
                              shrink_to_fit=False,
                              sort_queries=True)
    lbvh.build(pts_cupy)
    lbvh.prepare_knn_default(K, radius=None)
    idx_cupy, d2_cupy, nn_cupy = lbvh.query_knn(pts_cupy)
    idx_tensor = torch.as_tensor(idx_cupy.astype(cupy.int64), device=device)[None]  # !: BATCH
    d2_tensor = torch.as_tensor(d2_cupy, device=device)[None]
    nn_tensor = torch.as_tensor(nn_cupy.astype(cupy.int64), device=device)[None]

    torch.testing.assert_allclose(d2, d2_tensor)
    torch.testing.assert_allclose(idx, idx_tensor)


def raw_cupy_knn_points(p1: torch.Tensor,
                        p2: torch.Tensor,
                        K: int = 1,
                        return_nn: bool = True,
                        return_sorted: bool = True,

                        leaf_size=32,
                        compact=False,
                        shrink_to_fit=False,
                        radius=None,

                        return_lbvh=False,
                        ):
    import cupy
    import cupy_knn

    # !: BATCH
    p1_cupy = cupy.asarray(p1)
    p2_cupy = cupy.asarray(p2)
    lbvh = cupy_knn.LBVHIndex(leaf_size=leaf_size,
                              compact=compact,
                              shrink_to_fit=shrink_to_fit,
                              sort_queries=return_sorted)
    lbvh.build(p1_cupy)
    lbvh.prepare_knn_default(K, radius=radius)
    idx_cupy, d2_cupy, nn_cupy = lbvh.query_knn(p2_cupy)
    idx_tensor = torch.as_tensor(idx_cupy.astype(cupy.int64), device=p1.device)[None]  # !: BATCH
    d2_tensor = torch.as_tensor(d2_cupy, device=p1.device)[None]
    nn_tensor = torch.as_tensor(nn_cupy.astype(cupy.int64), device=p1.device)[None]

    if not return_lbvh:
        return d2_tensor, idx_tensor, nn_tensor
    else:
        return d2_tensor, idx_tensor, nn_tensor, return_lbvh


def test_cupy_knn_points_results():
    d2_0, idx_0, nn_0 = knn_points(pts, pts, K=K, return_nn=False, return_sorted=True)
    d2_1, idx_1, nn_1 = raw_cupy_knn_points(pts, pts, K=K, return_nn=False, return_sorted=True)

    torch.testing.assert_allclose(d2_0, d2_1)
    torch.testing.assert_allclose(idx_0, idx_1)


repeat = 500


def caching_lbvh_constructor(
    # These are required
    p1: torch.Tensor,
    return_sorted: bool = True,
    leaf_size=32,
    compact=False,
    shrink_to_fit=False,

    # These are dynamic
    K: int = 1,
    radius=None,
):
    key = (p1.data_ptr(), return_sorted, leaf_size, compact, shrink_to_fit)
    if key not in caching_lbvh_constructor.cache:
        import cupy
        import cupy_knn
        p1_cupy = cupy.asarray(p1)
        lbvh = cupy_knn.LBVHIndex(leaf_size=leaf_size,
                                  compact=compact,
                                  shrink_to_fit=shrink_to_fit,
                                  sort_queries=return_sorted)
        lbvh.build(p1_cupy)
        caching_lbvh_constructor.cache[key] = lbvh

        if len(caching_lbvh_constructor.cache) > caching_lbvh_constructor.maxsize:
            # caching_lbvh_constructor.cache.popitem(last=False)
            # Updated answer
            # In the context of the question, we are dealing with pseudocode, but starting in Python 3.8, := is actually a valid operator that allows for assignment of variables within expressions:
            # https://stackoverflow.com/questions/26000198/what-does-colon-equal-in-python-mean
            (k := next(iter(caching_lbvh_constructor.cache)), caching_lbvh_constructor.cache.pop(k))
    else:
        lbvh = caching_lbvh_constructor.cache[key]

    lbvh.prepare_knn_default(K, radius=radius)
    return lbvh


caching_lbvh_constructor.maxsize = 128
caching_lbvh_constructor.cache = dotdict()


def cupy_knn_points(p1: torch.Tensor,
                    p2: torch.Tensor,
                    K: int = 1,
                    return_nn: bool = True,
                    return_sorted: bool = True,

                    leaf_size=1024,
                    compact=False,
                    shrink_to_fit=False,
                    radius=None,

                    return_lbvh=False,
                    ):
    import cupy
    import cupy_knn
    # !: BATCH
    lbvh = caching_lbvh_constructor(p1, leaf_size, compact, shrink_to_fit, return_sorted, K, radius)

    p2_cupy = cupy.asarray(p2)
    idx_cupy, d2_cupy, nn_cupy = lbvh.query_knn(p2_cupy)
    idx_tensor = torch.as_tensor(idx_cupy.astype(cupy.int64), device=p1.device)[None]  # !: BATCH
    d2_tensor = torch.as_tensor(d2_cupy, device=p1.device)[None]
    nn_tensor = torch.as_tensor(nn_cupy.astype(cupy.int64), device=p1.device)[None]

    if not return_lbvh:
        return d2_tensor, idx_tensor, nn_tensor
    else:
        return d2_tensor, idx_tensor, nn_tensor, return_lbvh


def test_cupy_knn_pytorch3d_speed():
    torch.cuda.synchronize()
    timer.record('')
    for i in range(repeat):
        d2_0, idx_0, nn_0 = knn_points(pts, pts, K=K, return_nn=True, return_sorted=False)
    torch.cuda.synchronize()
    timer.record(f'PyTorch3D x {repeat}')

    for i in range(repeat):
        d2_0, idx_0, nn_0 = raw_cupy_knn_points(pts, pts, K=K, return_nn=True, return_sorted=True)
    torch.cuda.synchronize()
    timer.record(f'cupy-knn raw x {repeat}')

    for i in range(repeat):
        d2_0, idx_0, nn_0 = cupy_knn_points(pts, pts, K=K, return_nn=True, return_sorted=True)
    torch.cuda.synchronize()
    timer.record(f'cupy-knn cache x {repeat}')


if __name__ == '__main__':
    my_tests(globals())
