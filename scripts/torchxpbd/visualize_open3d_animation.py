import sys
import time
import numpy as np
import open3d as o3d

input_file = 'tshirt-drop#.npz' if len(sys.argv) <= 1 else sys.argv[1]
fps = 60 if len(sys.argv) <= 2 else float(sys.argv[2])
n_frame = 16384 if len(sys.argv) <= 3 else int(sys.argv[3])
animated_mesh = np.load(input_file)

faces = animated_mesh['faces']
animation = animated_mesh['animation'][:n_frame]
n_frame = len(animation)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(animation[0])
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()

origin = animation[0][145]
mesh_start = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=origin)
mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='XPBD + StVK + Bend Simulation', width=1920, height=1920)
vis.add_geometry(mesh)
vis.add_geometry(mesh_start)
vis.add_geometry(mesh_origin)
vis.poll_events()
vis.update_renderer()

opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.mesh_show_back_face = True
opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal

# i = 0
total_time = n_frame / fps
print(total_time)
start = time.time()

while True:
    curr = (time.time() - start) % total_time
    flt = curr / total_time * n_frame
    idx = int(flt)
    flt = flt - idx

    mesh.vertices = o3d.utility.Vector3dVector(animation[idx] * (1 - flt) + animation[(idx + 1) % n_frame] * flt)
    mesh.compute_vertex_normals()
    vis.update_geometry(mesh)

    vis.poll_events()
    vis.update_renderer()
