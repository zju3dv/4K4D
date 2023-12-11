import json
import torch
import struct
import turbojpeg
import numpy as np
from scipy.spatial.transform import Rotation

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict


def parse_unity_msg(msg):
    """ Parse the message from Unity. For now, we only support JSON format messages,
        with keys: 'signal', 'positionL', 'quaternionL', 'positionR', 'quaternionR'.
    Args:
        msg: the message from Unity, in a JSON format.
    Return:
        parsed_data: (dotdict), the parsed data from Unity.
    """
    try:
        parsed_data = json.loads(msg)
    except json.JSONDecodeError:
        log(red('Failed to parse the message from Unity.'))
    return dotdict(parsed_data)


def quat_tran_to_mat(quat: List[float], tran: List[float]):
    """ Convert the quaternion vector and translation vector into a 4x4 matrix.
        Can be used to convert to both w2c and c2w, depending on the input order.
    Args:
        quat: (np.ndarray), (4,), quaternion vector.
        tran: (np.ndarray), (3,), translation vector.
    Return:
        mat: (np.ndarray), (4, 4), the 4x4 matrix.
    """
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(quat).as_matrix()
    mat[:3, 3:] = np.array(tran)[..., None]
    return mat


def c2w_unity2opencv(c2w: np.ndarray):
    """ Convert the Unity camera2world to OpenCV camera2world.
    Args:
        c2w: (np.ndarray), (4, 4), camera to world matrix in Unity's OpenGL convention.
    Returns:
        c2w: (np.ndarray), (4, 4), camera to world matrix in easyvolcap's OpenCV convention.
    """
    c2w = c2w.copy()
    c2w[[0, 2], :] = -c2w[[0, 2], :]  # x-axis and z-axis reversal
    c2w = c2w[[0, 2, 1, 3], :]  # y-axis and z-axis switch
    # FIXME: I still do not know why we only need to inverse the y-axis here, but not the z-axis.
    # FIXME: As far as I know, the z-axis should be inversed as well (OpenGL to OpenCV obvious).
    # FIXME: But it has been proved in Unity `camera.WorldToViewportPoint()` func.
    c2w[0:3, 1:2] *= -1  # OpenGL to OpenCV convention
    return c2w


def trans_unity2opencv(t: np.ndarray):
    """ Convert the Unity translation to OpenCV translation.
    Args:
        t: (np.ndarray), (3,), translation vector in Unity's OpenGL convention.
    Returns:
        t: (np.ndarray), (3,), translation vector in easyvolcap's OpenCV convention.
    """
    t = t.copy()
    t[[0, 2]] = -t[[0, 2]]  # x-axis and z-axis reversal
    t = t[[0, 2, 1]]  # y-axis and z-axis switch
    return t


def compute_c2w_view_matrix(t: np.ndarray, o: np.ndarray):
    """ Compute the camera to world matrix from the translation vector.
        Assuming the camera is looking at the origin.
    Args:
        t: (np.ndarray), (3,), coordinate of the camera.
        o: (np.ndarray), (3,), coordinate of the looking center.
    Returns:
        c2w: (np.ndarray), (4, 4), camera to world matrix.
    """
    # Compute the camera's z-axis
    z = -(t - o)  # (3,)
    z = z / np.linalg.norm(z)  # (3,)
    # Compute the camera's x-axis
    x = np.cross(np.array([0, 1, 0]), z)  # (3, 1)
    x = x / np.linalg.norm(x)  # (3, 1)
    # Compute the camera's y-axis
    y = np.cross(z, x)  # (3, 1)
    c2w = np.eye(4)  # (4, 4)
    c2w[:3, :3] = np.stack([x, y, z], axis=1)
    c2w[:3, 3] = t
    return c2w


def unity_qt2opencv_w2c(quanternion: np.ndarray,
                        translation: np.ndarray,
                        scale: float = 1.0,
                        ):
    """ Decode the quaternion vector and translation vector from Unity into w2c.
        This is a simplest conversion function only changes from Unity to Easyvolcap.
    Args:
        translation: (np.ndarray), (3,), translation vector from Unity.
        quaternion: (np.ndarray), (4,), quaternion vector from Unity.
        scale: (float), the scale of the scene.
    Return:
        w2c: (torch.Tensor), (4, 4), world to camera matrix.
    """
    # Convert the quaternion vector and translation vector into a 4x4 c2w first
    c2w = quat_tran_to_mat(quanternion, translation)  # (4, 4)
    c2w[:3, 3:] *= scale
    # Then, perform camera-world convention transformation
    c2w = c2w_unity2opencv(c2w)  # (4, 4)
    # Convert c2w into w2c
    w2c = np.linalg.inv(c2w)  # (4, 4)
    return w2c


# # A more complex version, considering the coordinate system conversion and two center alignment.
# def decode_single_unity_pose(quanternion: np.ndarray,
#                              translation: np.ndarray,
#                              transformation: np.ndarray,
#                              origin: np.ndarray = None,
#                              scale: float = 1.0,
#                              ):
#     """ Decode the quaternion vector and translation vector from Unity into w2c.
#     Args:
#         translation: (np.ndarray), (3,), translation vector from Unity.
#         quaternion: (np.ndarray), (4,), quaternion vector from Unity.
#         transformation: (np.ndarray), (4, 4), affine transformation matrix to align the srd and the scene.
#         origin: (np.ndarray), (3,), the coordinate of the looking center.
#         scale: (float), the scale of the scene.
#     Return:
#         w2c: (torch.Tensor), (4, 4), world to camera matrix.
#     """
#     # Convert the quaternion vector and translation vector into a 4x4 c2w first
#     c2w = quat_tran_to_mat(quanternion, translation)  # (4, 4)
#     c2w[:3, 3:] *= scale
#     # Then, perform camera-world convention transformation
#     c2w = c2w_unity2opencv(c2w)  # (4, 4)
#     # Finally, align them with the current scene
#     c2w = transformation @ c2w  # (4, 4)
#     # Convert the c2w into w2c
#     w2c = torch.as_tensor(np.linalg.inv(c2w)).float()  # (4, 4)
#     return w2c


# # I mean, it should be the right version, considering coordinates convention and scene scaling.
# def decode_single_unity_pose(quanternion: np.ndarray,
#                              translation: np.ndarray,
#                              transformation: np.ndarray,
#                              origin: np.ndarray,
#                              scale: float = 1.0,
#                              ):
#     """ Decode the quaternion vector and translation vector from Unity into w2c.
#     Args:
#         translation: (np.ndarray), (3,), translation vector from Unity.
#         quaternion: (np.ndarray), (4,), quaternion vector from Unity.
#         transformation: (np.ndarray), (3,), relative translation vector to align the srd and the scene.
#         origin: (np.ndarray), (3,), the coordinate of the looking center.
#         scale: (float), the scale of the scene.
#     Return:
#         w2c: (torch.Tensor), (4, 4), world to camera matrix.
#     """
#     # Convert the translation vector into easyvolcap convention first
#     t = trans_unity2opencv(np.array(translation))  # (3,)
#     # Move the center of the srd tracking to the center of the scene
#     t = t + transformation  # (3,)
#     # Compute the c2w matrix
#     c2w = compute_c2w_view_matrix(t, origin)  # (4, 4)
#     # Convert the c2w into w2c
#     w2c = torch.as_tensor(np.linalg.inv(c2w)).float()  # (4, 4)
#     return w2c


# A version from the w2c perspective, which should make the easyvolcap
# scene is in the center of the unity
def decode_single_unity_pose(quanternion: np.ndarray,
                             translation: np.ndarray,
                             transformation: np.ndarray,
                             origin: np.ndarray = None,
                             scale: float = 1.0,
                             ):
    """ Decode the quaternion vector and translation vector from Unity into w2c.
    Args:
        translation: (np.ndarray), (3,), translation vector from Unity, only used to get original w2c.
        quaternion: (np.ndarray), (4,), quaternion vector from Unity.
        transformation: (np.ndarray), (4, 4), no use in this version of decoding function.
        origin: (np.ndarray), (3,), the fixed w2c translation vector.
        scale: (float), the scale of the scene.
    Return:
        w2c: (torch.Tensor), (4, 4), world to camera matrix.
    """
    # Perform coordinates conversion first
    w2c = unity_qt2opencv_w2c(quanternion, translation, scale)
    # Fix the w2c translation vector
    w2c[:3, 3] = origin
    # `w2c @ transformation` is equal to `transformation @ Pw`
    w2c = w2c @ transformation
    # Convert the w2c into torch.Tensor
    w2c = torch.as_tensor(w2c).float()  # (4, 4)
    return w2c


def decode_single_unity_proj_mats(k1, k2, k3, k4, H, W):
    """ Decode the projection matrix of Unity to camera intrinsic of Easyvolcap.
    Args:
        k1: (float), the [0, 0] element of the projection matrix, should equals to 2 * fx / W.
        k2: (float), the [0, 2] element of the projection matrix, should equals to 1 - 2 * cx / W.
        k3: (float), the [1, 1] element of the projection matrix, should equals to 2 * fy / H.
        k4: (float), the [1, 2] element of the projection matrix, should equals to 2 * cy / H - 1.
    Returns:
        K: (torch.Tensor), (3, 3), the camera intrinsic matrix of Easyvolcap.
    """
    fx = k1 * W / 2.
    cx = (1 - k2) * W / 2.
    fy = k3 * H / 2.
    cy = (k4 + 1) * H / 2.
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
    return K


def rectify_stereo_w2c_offset(w2cL: torch.Tensor, w2cR: torch.Tensor,
                              tL: List[float], tR: List[float]):
    """ Rectify the stereo offset for the two w2cs with same translation
    Args:
        w2cL: (torch.Tensor), (4, 4), the world2camera extrinsic matrix of the left camera.
        w2cR: (torch.Tensor), (4, 4), the world2camera extrinsic matrix of the right camera.
        tL: (List[float]), the original Unity c2w translation of the left camera.
        tR: (List[float]), the original Unity c2w translation of the right camera.
    Returns:
        w2cR: (torch.Tensor), (4, 4), rectified world2camera extrinsic matrix of the right camera.
    """
    c2w_offset = np.array(tR) - np.array(tL)  # (3,)
    c2w_offset = torch.tensor(trans_unity2opencv(c2w_offset))[:, None].float()  # (3, 1)
    w2cR_trans = w2cR[:3, :3] @ (w2cL[:3, :3].T @ w2cL[:3, 3:] - c2w_offset)  # (3, 1)
    w2cR[:3, 3:] = w2cR_trans
    return w2cR


def decode_stereo_unity_poses(data, transformation: np.ndarray, origin: np.ndarray = None,
                              scale: float = 1.0, H: int = 2160, W: int = 3840):
    """ Decode the stereo poses from Unity into R, T respectively, by stereo poses
        we mean the left and right camera poses.
    Args:
        data: (dotdict), the parsed data which includs the left and right camera poses.
        transformation: (np.ndarray), (4, 4), affine transformation matrix to align the srd and the scene.
        origin: (np.ndarray), (3, 1), the coordinate of the looking center.
        scale: (float), the scale of the scene.
        H, W: (int, int), the height and width of the rendered image.
    Return:
        signal: (int), the signal from Unity, -1 means the end of the episode.
        w2cL: (torch.Tensor), (4, 4), the world2camera extrinsic matrix of the left camera.
        w2cR: (torch.Tensor), (4, 4), the world2camera extrinsic matrix of the right camera.
    """
    signal = int(data.signal)
    # print("quaternionL: ", data.quaternionL, ", positionL: ", data.positionL, ", projMatL: ", data.projMatL)
    # print("quaternionR: ", data.quaternionR, ", positionR: ", data.positionR, ", projMatR: ", data.projMatR)
    w2cL = decode_single_unity_pose(data.quaternionL, data.positionL, transformation, origin, scale)  # (4, 4)
    w2cR = decode_single_unity_pose(data.quaternionR, data.positionR, transformation, origin, scale)  # (4, 4)
    # # FIXME: cannot perform recify here since we fix the camera intrinsic (but SRD will adjust the camera intrinsic according to pose)
    # w2cR = rectify_stereo_w2c_offset(w2cL, w2cR, data.positionL, data.positionR)  # (4, 4)
    KL = decode_single_unity_proj_mats(*(data.projMatL), H, W)
    KR = decode_single_unity_proj_mats(*(data.projMatR), H, W)
    return signal, w2cL, w2cR, KL, KR


def unity_qt2w2c(quaternion: np.ndarray, translation: np.ndarray, scale: float = 1.0):
    """ Convert the quaternion vector and translation vector from Unity into w2c.
    Args:
        quaternion: (np.ndarray), (4,), quaternion vector from Unity.
        translation: (np.ndarray), (3,), translation vector from Unity.
        scale: (float), the scale of the scene.
    Return:
        w2c: (np.ndarray), (4, 4), world to camera matrix.
    """
    # Convert the quaternion vector and translation vector into a 4x4 c2w first
    c2w = quat_tran_to_mat(quaternion, translation)  # (4, 4)
    c2w[:3, 3:] *= scale
    # Convert the c2w into w2c
    w2c = np.linalg.inv(c2w)  # (4, 4)
    return w2c


def RT2c2w(R, T):
    """ Convert the rotation matrix and translation matrix into camera to world matrix.
    Args:
        R: (torch.Tensor), (3, 3), rotation matrix.
        T: (torch.Tensor), (3, 1), translation matrix.
    Return:
        c2w: (torch.Tensor), (4, 4), camera to world matrix.
    """
    w2c = torch.eye(4, device=R.device)
    w2c[:3, :3], w2c[:3, 3:] = R, T
    c2w = torch.inverse(w2c)
    return c2w


def encode_easyvolcap_stereo_imgs(tensor_left: torch.Tensor, tensor_right: torch.Tensor):
    """ Encode the easyvolcap rendered stereo image pairs using `turbojpeg`, which is
        the fastest encoding method I know so far, the encoded bytes stream is in structured
        as 4 bytes(encoded bytes number of the left image) + encoded left image + encoded
        right image.
    Args:
        tensor_left: (torch.Tensor), (B, H, W, 4), rendered left eye RGBA image, already detach and on cpu.
        tensor_right: (torch.Tensor), (B, H, W, 4), rendered right eye RGBA image, already detach and on cpu.
    Returns:
        bytes stream including 4 bytes of left encoded image, encoded left and right image.
    """
    jpeg = turbojpeg.TurboJPEG()
    # Encode the rendered images for left eye and right respectively
    bytes_left = jpeg.encode(tensor_left.numpy(), quality=70, pixel_format=turbojpeg.TJPF_RGBA)
    bytes_right = jpeg.encode(tensor_right.numpy(), quality=70, pixel_format=turbojpeg.TJPF_RGBA)
    # Need to tell C++ the length of the first encoded image to avoid unnecessary decoding
    length_left = struct.pack('i', len(bytes_left))
    length_right = struct.pack('i', len(bytes_right))
    return length_left + length_right + bytes_left + bytes_right


def log_stereo_params(w2cL, w2cR, KL, KR):
    """ Print converted w2c of the left eye and the right eye to the terminal
    Args:
        w2cL: (torch.Tensor), (4, 4), the world to camera matrix of the left eye
        w2cR: (torch.Tensor), (4, 4), the world to camera matrix of the right eye
        KL: (torch.Tensor), (3, 3), the camera intrinsic matrix of the left eye
        KR: (torch.Tensor), (3, 3), the camera intrinsic matrix of the right eye
    Returns:
        None
    """
    # Just to make the main code looks better
    print(w2cL)
    print(w2cR)
    print(KL)
    print(KR)
