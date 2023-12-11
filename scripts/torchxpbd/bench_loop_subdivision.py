import time
import torch
from tqdm import tqdm

# fmt: off
import sys
sys.path.append('.')

from easyvolcap.utils.data_utils import load_mesh, export_mesh
from easyvolcap.utils.mesh_utils import triangle_to_halfedge, halfedge_to_triangle, multiple_halfedge_loop_subdivision
# fmt: on


depth = 2
repeat = 1000
input_file = 'big-sigcat.ply'
output_file = 'big-sigcat_2.ply'
v, f = load_mesh(input_file)
he = triangle_to_halfedge(v, f)

print(f'vert count: {he.V}')
print(f'face count: {he.F}')
print(f'edge count: {he.E}')
print(f'halfedge count: {he.HE}')

rounds = []
for i in tqdm(range(repeat)):
    start = time.time()
    he = triangle_to_halfedge(v, f, False)
    torch.cuda.synchronize()
    end = time.time()
    rounds.append(end - start)

print('Conversion from triangle to halfedge representation: ')
print(f'fastest time: {min(rounds) * 1000:.3f}ms')
print(f'slowest time: {max(rounds) * 1000:.3f}ms')
print(f'average time: {sum(rounds) / len(rounds) * 1000:.3f}ms')

rounds = []
for i in tqdm(range(repeat)):
    start = time.time()
    he = triangle_to_halfedge(v, f, True)
    torch.cuda.synchronize()
    end = time.time()
    rounds.append(end - start)

print('Conversion from triangle to halfedge with manifold assumption: ')
print(f'fastest time: {min(rounds) * 1000:.3f}ms')
print(f'slowest time: {max(rounds) * 1000:.3f}ms')
print(f'average time: {sum(rounds) / len(rounds) * 1000:.3f}ms')

rounds = []
for i in tqdm(range(repeat)):
    start = time.time()
    he = triangle_to_halfedge(v, f, True)
    end = time.time()
    rounds.append(end - start)
torch.cuda.synchronize()

print('Conversion from triangle to halfedge with manifold assumption (no gpu sync): ')
print(f'fastest time: {min(rounds) * 1000:.3f}ms')
print(f'slowest time: {max(rounds) * 1000:.3f}ms')
print(f'average time: {sum(rounds) / len(rounds) * 1000:.3f}ms')

rounds = []
for i in tqdm(range(repeat)):
    start = time.time()
    nhe = multiple_halfedge_loop_subdivision(he, depth, False)
    torch.cuda.synchronize()
    end = time.time()
    rounds.append(end - start)

print(f'Torch Loop subdivision (depth: {depth}): ')
print(f'fastest cpu time: {min(rounds) * 1000:.3f}ms')
print(f'slowest time: {max(rounds) * 1000:.3f}ms')
print(f'average time: {sum(rounds) / len(rounds) * 1000:.3f}ms')

rounds = []
for i in tqdm(range(repeat)):
    start = time.time()
    nhe = multiple_halfedge_loop_subdivision(he, depth, True)
    torch.cuda.synchronize()
    end = time.time()
    rounds.append(end - start)

print(f'Torch Loop subdivision with manifold assumption (depth: {depth}): ')
print(f'fastest time: {min(rounds) * 1000:.3f}ms')
print(f'slowest time: {max(rounds) * 1000:.3f}ms')
print(f'average time: {sum(rounds) / len(rounds) * 1000:.3f}ms')

rounds = []
for i in tqdm(range(repeat)):
    start = time.time()
    nhe = multiple_halfedge_loop_subdivision(he, depth, True)
    end = time.time()
    rounds.append(end - start)

torch.cuda.synchronize()
print(f'Torch Loop subdivision with manifold assumption (depth: {depth}) (no gpu sync): ')
print(f'fastest cpu time: {min(rounds) * 1000:.3f}ms')
print(f'slowest time: {max(rounds) * 1000:.3f}ms')
print(f'average time: {sum(rounds) / len(rounds) * 1000:.3f}ms')

# ~1.566ms for 2 steps on 3090
# while a loop subdiv on MeshLab takes ~232ms

# ~60ms for 5 steps
# while MeshLab takes 17991ms

v, f = halfedge_to_triangle(nhe)
export_mesh(v, f, filename=output_file)
