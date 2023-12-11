# fmt: off
try:
    import ipdb
    import numpy as np
    from tqdm import tqdm
    from termcolor import colored
except ImportError:
    import sys
    import subprocess
    # Defaulting to user installation because normal site-packages is not writeable...
    # from os.path import join
    # site_packages = join(sys.prefix, 'lib', 'site-packages')
    subprocess.call([sys.executable, '-m', 'ensurepip'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'pip',                        ])
    subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'wheel',                      ])
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'numpy', 'tqdm', 'ipdb', 'termcolor'])

import bpy
import sys
import numpy as np
from tqdm import tqdm
from termcolor import colored
from typing import List, Dict
# fmt: on


def log(msg, color=None, attrs=None):
    func = sys._getframe(1).f_code.co_name
    frame = sys._getframe(1)
    module = frame.f_globals['__name__'] if frame is not None else ''
    tqdm.write(colored(module, 'blue') + " -> " + colored(func, 'green') + ": " + colored(str(msg), color, attrs))  # be compatible with existing tqdm loops


"""
import os
import sys
module_path = 'z:/dngp/lib/utils/blender_utils.py'
file_path = f"z:/dngp/data/pro/A/shape_fitting_results/optim_tpose.npz"
module_dir = os.path.dirname(module_path)
sys.path.append(module_dir)
from blender_utils import *
# replace_weights(file_path, file_path)
load_rigged(file_path, load_bw=True)

import numpy as np
joints = bpy.data.objects['joints']
keyframes = np.load('z:/dngp/motion_sithand_repair.npz')
assign_keyframes(joints, keyframes)
"""

"""
blender --background --python tools/genBW.py -- --file_path data/pro/A/optim_mesh.npz --target_file_path data/pro/A/blend_mesh.npz


all_rs, all_ts, all_poses = item[:, 1, :], item[:, 0, :], np.concatenate([np.zeros_like(item[:, :1, :]), item[:, 2:, :]], axis=1)
np.savez_compressed('motion1_proxjj_repair_smoothing.npz', all_rs=all_rs, all_ts=all_ts, all_poses=all_poses)
"""


def replace_weights(file_path="c:/users/aaa/desktop/can_mesh.npz", target_path="c:/users/aaa/desktop/can_mesh.npz"):
    mesh, joints, item = load_rigged(file_path, load_bw=False, root_and_pin=False, align_bone_to_y=False, joints_name='joints')
    automatic_weights(mesh, joints)
    bone_names = [f'{i}' for i in range(len(item['joints']))]
    log(f'using valid bone names: {bone_names}')
    weights = extract_weights(mesh, bone_names)
    item = {**item}
    item['weights'] = weights.astype(np.float32)
    log(f'saving data with blend weights: {target_path}')
    np.savez_compressed(target_path, **item)


def automatic_weights(mesh: bpy.types.Object, joints: bpy.types.Object):
    log('generating automatic blend weights')
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')  # deselect all object
    mesh.select_set(True)
    joints.select_set(True)  # select the object for the 'parenting'
    bpy.context.view_layer.objects.active = joints  # the active object will be the parent of all selected object
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')


def load_rigged(file_path="c:/users/aaa/desktop/can_mesh.npz", mesh_name='mesh', joints_name='tjoints', load_bw=True, load_bs=False, **kwargs):
    log(f"loading data from: {file_path}")
    item = np.load(file_path, allow_pickle=True)

    log(f"adding mesh to blender")
    verts = item.get('verts', item.get('vertices', None))
    faces = item.get('faces', item.get('triangles', None))
    uv = item.get('uv', None)
    uvfaces = item.get('uvfaces', None)
    img = item.get('img', None)
    mesh = add_mesh(mesh_name, verts, faces, uv, uvfaces, img)
    parents = item['parents']
    parents[0] = -1  # FIXME: manually setting first as root
    location = item.get(joints_name)

    log(f"adding joints to blender")
    joints = add_joints(joints_name, mesh, location, parents, **kwargs)

    if load_bw:
        log(f"adding blend weights to blender")
        if 'bones' in item:
            bone_ids = item['bones']
        else:
            bone_ids = np.arange(len(location))[None].repeat(len(verts), axis=0)
        weights = item.get('weights', item.get('weights', None))
        weights /= weights.sum(axis=-1, keepdims=True)
        assign_bw(mesh, weights, bone_ids)

    if load_bs:
        log(f"adding blend shapes to blender")
        # one confusinog question:
        # 1. sometimes, different bones have different number of datapoints
        #    this means we'd have to split the implementation to different blend shape groups
        #    stored in separate files, so this means shared data should be loaded in `add_driver_function`
        #    and bone & indices specific data should be loaded in `assign_blend_shape`
        for grp_idx, params in enumerate(item['blend_params']):
            add_drive_function(params['matrix_dict'], params['posedirs'], params['bonedirs'], params['blend_poses'], params['blend_bones'], params['use_bones'], params['use_parent'], grp_idx)
            assign_blend_shape(mesh, joints, item['parents'], params['blends'], params['blend_bones'], grp_idx)

    return mesh, joints, item


def update_expression(d: bpy.types.FCurve):
    # https://blender.stackexchange.com/questions/118350/how-to-update-the-dependencies-of-a-driver-via-python-script
    d.driver.expression += " "
    d.driver.expression = d.driver.expression[:-1]


def update_dependencies(ob: bpy.types.Object):
    if ob.animation_data is not None:
        drivers = ob.animation_data.drivers
        log(drivers)
        if 'drivers':
            for d in drivers:
                log(d)
                update_expression(d)
    for sk_set in bpy.data.shape_keys:
        for d in sk_set.animation_data.drivers:
            log(f'updating driver for: {d}')
            update_expression(d)


def update_all_driver_dependencies():
    for ob in bpy.data.objects:
        update_dependencies(ob)


def add_quaternion_rotation_driver(sk: bpy.types.Object, joints: bpy.types.Object, pb: int, b: int, d: int, bi: int, grp_idx: int = 0):
    dims = ['x', 'y', 'z', 'w']
    rot_dims = ['ROT_X', 'ROT_Y', 'ROT_Z', 'ROT_W']

    driver: bpy.types.Driver = sk.driver_add('value').driver
    driver.type = 'SCRIPTED'
    # using parent space bone direction
    driver.expression = f'''driver_interpolate_grp{grp_idx}(px=px, py=py, pz=pz, pw=pw, x=x, y=y, z=z, w=w, bi={bi}, d={d})'''

    for rot_v, rot_d in zip(dims, rot_dims):
        var = driver.variables.new()
        var.name = rot_v
        var.type = 'TRANSFORMS'
        tar = var.targets[0]
        tar.id = joints
        tar.bone_target = f'{b}'
        tar.rotation_mode = 'QUATERNION'
        tar.transform_space = 'LOCAL_SPACE'
        tar.transform_type = rot_d

    for rot_v, rot_d in zip(dims, rot_dims):
        var = driver.variables.new()
        var.name = f'p{rot_v}'
        var.type = 'TRANSFORMS'
        tar = var.targets[0]
        tar.id = joints
        tar.bone_target = f'{pb}'
        tar.rotation_mode = 'QUATERNION'
        tar.transform_space = 'LOCAL_SPACE'
        tar.transform_type = rot_d


def assign_blend_shape(mesh: bpy.types.Object, joints: bpy.types.Object, parents: list, shapes: np.ndarray, bone_ids: np.ndarray = None, grp_idx: int = 0):
    vertices = mesh.data.vertices
    dims = ['x', 'y', 'z', 'w']
    pbar = tqdm(range(shapes.shape[0] * shapes.shape[1]))

    if not mesh.data.shape_keys or 'basis' not in mesh.data.shape_keys.key_blocks:
        sk_basis = mesh.shape_key_add(name='basis')
        mesh.data.shape_keys.use_relative = True

    if bone_ids is not None:
        it = bone_ids
    else:
        it = range(shapes.shape[0])

    for bi, b in enumerate(it):  # bone
        for d in range(shapes.shape[1]):  # dim: x, y, z quaternion
            shape = shapes[bi, d]
            sk = mesh.shape_key_add(name=f'{b}_{d}')
            sk.slider_max = 1.0
            sk.slider_min = -1.0
            for i in tqdm(range(len(vertices))):
                sk.data[i].co = np.array(vertices[i].co) + shape[i]  # assign vertex location
            add_quaternion_rotation_driver(sk, joints, pb=parents[b], b=b, d=d, bi=bi, grp_idx=grp_idx)
            pbar.update(n=1)


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)
    rot_mat = rot_mat.astype(np.float32)
    return rot_mat


def add_drive_function(matrix_dict: dict[list, np.ndarray], posedirs: np.ndarray, bonedirs: np.ndarray, blend_poses: np.ndarray, blend_bones: np.ndarray, use_bones: bool = True, use_parent: bool = True, grp_idx: int = 0):
    # matrix_dict: mapping from [0, 1, 2] string of list to a 3x3 matrix, for solving the interpolation equation
    # posedirs:    the data point of our interpolation, posed bone directions
    # bonedirs:    the starting points of our transformation, when applying current pose, bone directions will change, and the changed bone direction will be used as interpolation target for the posedirs (whose inverse has been computed in advance for solving the system of linear equations)
    # blend_poses: the raw poses as data points (not used in interpolation, only for visualization now)

    def quaternion_to_angle_axis(x: float, y: float, z: float, w: float):
        if w == 1:
            return np.array([x, y, z])
        angle = 2 * np.arccos(w)
        angle_axis = np.array([x, y, z]) / np.sin(angle / 2) * angle  # quaternion to angle axis
        return angle_axis

    def apply_rotation(x: float, y: float, z: float, w: float, source: np.ndarray):
        # quaternion to angle axis
        # if w == 1, arccos gives a zero angle, which will lead to error when doing division

        angle_axis = quaternion_to_angle_axis(x, y, z, w)
        R = batch_rodrigues(angle_axis[None])[0]
        target = R @ source  # NOTE: loses one DOF

        return target

    def driver_interpolate(
        bi: int, d: int,
        x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0,
        px: float = 0.0, py: float = 0.0, pz: float = 0.0, pw: float = 1.0,  # if not passed, only applying one level of rotation
    ):
        # xp: x of quaternion of parent bone
        # yp: y of quaternion of parent bone
        # zp: z of quaternion of parent bone
        # wp: w of quaternion of parent bone
        # x:  x of quaternion of current bone
        # y:  y of quaternion of current bone
        # z:  z of quaternion of current bone
        # w:  w of quaternion of current bone
        # bi: bone id of added blend shapes (like 0, 1)
        # d:  dimension of the activated blend shape (TODO: yes, duplicate computation...)

        target = quaternion_to_angle_axis(x, y, z, w)
        if use_bones:
            target = apply_rotation(x, y, z, w, bonedirs[bi])
            if use_parent:
                target = apply_rotation(px, py, pz, pw, target)

        # select 3 closest
        # dists: np.ndarray = ((batch_rodrigues(optim_poses[bi]) - batch_rodrigues(target[None])) ** 2).sum(axis=-1).sum(axis=-1)
        dists: np.ndarray = 1 - (posedirs[bi] * target[None]).sum(axis=-1)  # 1 - dot product (cos)

        indices = dists.argsort()[:3]
        indices.sort()
        indices: list = indices.tolist()

        if d not in indices:
            return 0

        log(f'grp_id: {colored(grp_idx, "red")}')
        log(f'bone_id: {colored(bi, "red")}')
        log(f'bone_name: {colored(blend_bones[bi], "red")}')
        log(f'shape_id: {colored(d, "red")}')
        log(f'dists: {dists}')

        matrix = matrix_dict[str(indices)]

        # target = np.array([x, y, z])
        weights = matrix[bi] @ target  # fitting
        # weights = weights / (np.linalg.norm(weights) + 1e-13)
        # source = source.clip(0, 1)
        # source /= source.sum() + eps

        fitted = posedirs[bi][indices].T @ weights
        log(f'selected:\n{posedirs[bi][indices]}')
        log(f'indices: {colored(indices, "magenta")}')

        angles = np.linalg.norm(blend_poses[bi][indices], axis=-1)
        axes = blend_poses[bi][indices] / angles
        log(f'selected angles:\n{colored(angles, "cyan")}')
        log(f'selected axes:\n{colored(axes, "cyan")}')

        log(f'weights: {colored(weights, "green")}')
        log(f'fitted: {colored(fitted, "green")}')
        log(f'target: {colored(target, "green")}')

        weight = weights[indices.index(d)]
        return weight

    bpy.app.driver_namespace[f'driver_interpolate_grp{grp_idx}'] = driver_interpolate    # add function to driver_namespace
    update_all_driver_dependencies()


def extract_weights(mesh: bpy.types.Object, bone_names=[]):
    """
    export skin weights on selected character
    """
    vertex_groups = mesh.vertex_groups
    vertices = mesh.data.vertices
    vertex_ids = [v.index for v in vertices]

    # assuming bone names to go up like 0, 1, 2...
    weights = np.zeros(shape=(len(vertex_ids), len(vertex_groups) if not bone_names else len(bone_names)))

    for vertex_id in tqdm(vertex_ids):
        for grp in vertices[vertex_id].groups:
            if bone_names:
                vertex_group_name = vertex_groups[grp.group].name
                if vertex_group_name in bone_names:
                    bone = bone_names.index(vertex_group_name)
                else:
                    continue  # filter
            else:
                bone = grp.group
            weight = grp.weight
            weights[int(vertex_id), int(bone)] = weight

    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights


def assign_keyframes(joints: bpy.types.Object, keyframes: dict):
    pose_bones = joints.pose.bones
    pin = pose_bones['pin']
    fix_transform(0, pin)

    all_poses, all_rs, all_ts = keyframes['all_poses'], keyframes['all_rs'], keyframes['all_ts']
    indices = list(range(len(all_poses)))
    for frame_id in tqdm(indices):
        insert_keyframe(frame_id, all_poses[frame_id], all_rs[frame_id], all_ts[frame_id], pose_bones)
    # prepare a scene
    scn = bpy.context.scene
    scn.frame_start = indices[0]
    scn.frame_end = indices[-1]


def insert_keyframe(frame_id, poses, Rh, Th, pose_bones: bpy.types.bpy_prop_collection):
    #    poses = R.from_rotvec(poses).inv().as_rotvec()
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    axis = poses / angle
    axis_angle = np.concatenate([angle, axis], axis=-1)

    Rh_angle = np.linalg.norm(Rh + 1e-8, keepdims=True)
    Rh_axis = Rh / Rh_angle
    Rh_axis_angle = np.concatenate([Rh_angle, Rh_axis], axis=-1)

    for i in range(poses.shape[0]):
        pose_bone = pose_bones[str(i)]
        pose_bone.rotation_mode = 'AXIS_ANGLE'
        pose_bone.rotation_axis_angle = axis_angle[i]
        pose_bone.keyframe_insert('rotation_axis_angle', frame=frame_id)

    root = pose_bones['root']
    root.rotation_mode = 'AXIS_ANGLE'
    root.rotation_axis_angle = Rh_axis_angle
    root.location = Th
    root.keyframe_insert('rotation_axis_angle', frame=frame_id)
    root.keyframe_insert('location', frame=frame_id)


def fix_transform(frame_id, object: bpy.types.Object):
    object.rotation_mode = 'AXIS_ANGLE'
    object.keyframe_insert('rotation_axis_angle', frame=frame_id)
    object.keyframe_insert('location', frame=frame_id)
    object.keyframe_insert('scale', frame=frame_id)


def add_mesh(name, vs: np.ndarray, fs: np.ndarray, uv: np.ndarray = None, uvfaces: np.ndarray = None, img: np.ndarray = None):
    vs = vs.tolist()
    fs = fs.tolist()  # blender doesn't accept np array...
    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(vs, [], fs)
    mesh_data.update()

    mesh = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(mesh)

    if uv is not None and uvfaces is not None:
        log("found uv coordinates and uvfaces, loading...")
        # also add uv coordinates
        uvlayer = mesh_data.uv_layers.new()
        mesh_data.uv_layers.active = uvlayer
        for face in tqdm(mesh_data.polygons):
            for i, loop_idx in enumerate(face.loop_indices):
                uvlayer.data[loop_idx].uv = uv[uvfaces[face.index]][i]

    if img is not None:
        log("found texture images, loading...")
        # create empty image of blender
        has_alpha = img.shape[-1] == 4
        # Note: choose if the image should have alpha here, but even if
        # it doesn't, the array still needs to be RGBA
        image = bpy.data.images.new('texture.png', img.shape[1], img.shape[0], alpha=has_alpha)
        # fast way to set pixels (since 2.83)
        img_rgba_fp32 = img.astype(np.float32) if has_alpha else np.concatenate([img, np.ones((*img.shape[:2], 1))], axis=-1).astype(np.float32)
        img_rgba_fp32 = np.flip(img_rgba_fp32, axis=0)
        image.pixels.foreach_set(img_rgba_fp32.ravel())
        # pack the image into .blend so it gets saved with it

        # create new material
        mat = bpy.data.materials.new(name="mat")
        mat.use_nodes = True

        # create new texture
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = image

        # assign texture to material base color
        bsdf = mat.node_tree.nodes['Principled BSDF']
        surf = mat.node_tree.nodes['Material Output']
        mat.node_tree.nodes.remove(bsdf)
        emission = mat.node_tree.nodes.new('ShaderNodeEmission')
        mat.node_tree.links.new(tex.outputs['Color'], emission.inputs['Color'])
        mat.node_tree.links.new(emission.outputs['Emission'], surf.inputs['Surface'])

        # assign material to mesh
        mesh_data.materials.append(mat)

    return mesh


def add_joints(name, mesh: bpy.types.Object, location: np.ndarray, parents: np.ndarray, root_and_pin=True, align_bone_to_y=True):
    n_bones = len(parents)
    def is_root(p): return p < 0 or p >= n_bones

    children = [[i for i, a in enumerate(parents) if a == p] for p in range(len(parents))]  # a for ancestor

    joints_data = bpy.data.armatures.new(name)
    joints = bpy.data.objects.new(name, joints_data)
    bpy.context.collection.objects.link(joints)

    bpy.context.view_layer.objects.active = joints
    joints.show_in_front = True
    # must be in edit mode to add bones
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)

    edit_bones = joints.data.edit_bones

    if root_and_pin:
        pin = edit_bones.new('pin')
        pin.head = [0, -1, 0]
        pin.tail = [0, 0, 0]

        # WE WILL NEED TO CREATE A MASTER BONE, WHOSE TAIL IS THE CENTER JOINT OF OTHERS
        # Needs a root to support transformation
        root_children = [i for i, a in enumerate(parents) if is_root(a)]
        root = edit_bones.new('root')
        root.head = [0, 0, 0]
        length = np.linalg.norm(np.mean(location[root_children] - root.head, axis=0))
        root.tail = root.head
        root.tail[1] += length
        root.parent = pin

    for i in range(0, len(location)):
        b = edit_bones.new(str(i))
        b.head = location[i]  # set head location
        if is_root(parents[i]):
            if root_and_pin:
                b.parent = root
        else:
            # Assuming a topological sorted order
            pi = parents[i]  # parent index
            p = edit_bones[str(pi)]
            b.parent = p

        # !: Bone's local coordinate system is always gonna point in the Y axis direction
        # AND WE WANT THE GLOBAL AXIS TO ALIGH WITH THE LOCAL ONES
        # SINCE THE CODE IN anisdf IS LIKE SO...
        # But when loading in the bones for blender to generate automatic weights, we need to align from parent to child
        # Note that there's still gonna be issues about where those two child bones should be located...
        if len(children[i]) != 0:
            if align_bone_to_y:
                length = np.linalg.norm(np.mean(location[children[i]] - b.head, axis=0))
                b.tail = b.head
                b.tail[1] += length
            else:
                location_children = location[children[i]]
                b.tail = np.array(location_children).mean(axis=-2)
        else:
            if align_bone_to_y:
                length = np.linalg.norm(2 * location[i] - location[pi] - b.head)
                b.tail = b.head
                b.tail[1] += length
            else:
                b.tail = 2 * location[i] - location[pi]

        # !: Let's hope from_pydata won't change the indices
        g = mesh.vertex_groups.new(name=str(i))

    # exit edit mode to save bones so they can be used in pose mode
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh.parent = joints
    modifier = mesh.modifiers.new(type='ARMATURE', name="Armature")
    modifier.object = joints

    return joints


def assign_bw(mesh: bpy.types.Object, weights, bone_ids):
    # note that bones are marked by ids, should correspond with vertex group
    for v, bw in enumerate(tqdm(weights)):
        for i, bone_id in enumerate(bone_ids[v]):
            if bw[i] > 0:
                bone_name = str(bone_id)  # might already be a string
                if bone_name not in mesh.vertex_groups:
                    # !: Let's hope from_pydata won't change the indices
                    g = mesh.vertex_groups.new(name=bone_name)
                g = mesh.vertex_groups[bone_name]
                g.add([v], bw[i], 'REPLACE')