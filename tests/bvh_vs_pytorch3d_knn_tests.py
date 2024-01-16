from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.data_utils import load_pts, load_mesh
from easyvolcap.utils.chunk_utils import multi_gather_tris

from bvh_distance_queries import BVH
from pytorch3d.ops import knn_points

from easyvolcap.utils.test_utils import my_tests
bvh = BVH(queue_size=1024)

verts, faces = load_mesh('assets/meshes/bunny.ply', device='cuda')  # only the vertices matter
verts, faces = verts[None], faces[None]

repeat = 500
K = 10

def test_bvh_dq_vs_10_knn_points():
    timer.record('')
    for i in range(repeat):
        d2, idx, nn = knn_points(verts, verts, K=K, return_sorted=True, return_nn=False)
    timer.record(f'pytoch3d knn, K={K}, {repeat} x')

    for i in range(repeat):
        d2, pts, fids, bcs = bvh(multi_gather_tris(verts, faces), verts)
    timer.record(f'bvh dists, surafce, {repeat} x')

    tris = multi_gather_tris(verts, faces)
    for i in range(repeat):
        d2, pts, fids, bcs = bvh(tris, verts)
    timer.record(f'bvh dists, triangles, {repeat} x')


if __name__ == '__main__':
    my_tests(globals())
