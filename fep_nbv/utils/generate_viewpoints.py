import sys
import healpy as hp
import numpy as np
import tyro
import torch
import mathutils
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import quaternion_multiply, quaternion_apply

from fep_nbv.utils.transform_viewpoints import xyz2pose

def rotate_around_z(points, angle):
    """
    绕 z 轴旋转点集。
    
    :param points: 输入点集，形状为 (N, 3)
    :param angle: 旋转角度（弧度）
    :return: 旋转后的点集
    """
    rotation = R.from_euler('z', angle)
    return points @ rotation.as_matrix().T

def rotate_to_target(points, target=None):
    """
    将所有点绕球面旋转，使第一个点对齐到目标视角。
    :param points: 输入点集 (N, 3)
    :param target: 目标视角 (3,) 笛卡尔坐标
    :return: 旋转后的点集
    """
    # 归一化目标视角
    target = target / np.linalg.norm(target)

    # 第一个点
    first_point = points[0] / np.linalg.norm(points[0])

    # 计算旋转轴和旋转角度
    v = np.cross(first_point, target)  # 旋转轴
    s = np.linalg.norm(v)  # sin(θ)
    c = np.dot(first_point, target)   # cos(θ)

    # 如果点已经对齐，则不需要旋转
    if s == 0:
        return points

    # 反对称矩阵
    v_skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Rodrigues 旋转公式
    rotation_matrix = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / (s**2))
    return points @ rotation_matrix.T

# def generate_HEALPix_viewpoints(n_side = 2,radius = 2, original_viewpoint=None,offset_phi=0):
#     num_pixels = hp.nside2npix(n_side) # 计算总像素数
#     theta, phi = hp.pix2ang(n_side, np.arange(num_pixels)) # 获取每个像素的中心坐标 (θ, φ)
#     # phi = (phi+offset_phi)%(2*np.pi)

#     # 转换为笛卡尔坐标
#     x = radius*np.sin(theta) * np.cos(phi)
#     y = radius*np.sin(theta) * np.sin(phi)
#     z = radius*np.cos(theta)

#     points = np.stack([x, y, z], axis=-1)
#     points = rotate_to_target(points, np.array([0, 0, 1]))
#     points = rotate_around_z(points, offset_phi)

#     if original_viewpoint is None:
#         original_viewpoint = np.array([0, 0, 1])
#     points = rotate_to_target(points, original_viewpoint)

#     poses = xyz2pose(points[:,0],points[:,1],points[:,2])

#     # 对第一个pose进行旋转
#     axis = (points[0] / np.linalg.norm(points[0]))  # 获取旋转轴
#     if offset_phi==0:
#         rotation_quat = np.array([0,0,0,1])
#     else:
#         rotation_quat = R.from_rotvec(offset_phi * axis).as_quat()  # 生成旋转四元数
#     # rotation_quat = rotation_quat[[3, 0, 1, 2]] 
#     q_0 = (R.from_quat(poses[0,:4]) * R.from_quat(rotation_quat)).as_quat()
#     print(f'rotation: {R.from_rotvec(offset_phi * axis).apply(np.array([0,0,-1]))}')
#     print(f'origin: {R.from_quat(poses[0,:4]).apply(np.array([0,0,-1]))}')
#     print(f'q_0 : { (R.from_quat(poses[0,:4]) * R.from_quat(rotation_quat)).apply(np.array([0,0,-1]))}')
#     print(f'points {points[0]}')
#     poses[0,:4] = torch.Tensor(q_0)

#     return poses # xyzwxyz


def look_at_quaternion(position, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    """
    计算相机从 position 看向 target 时的四元数。
    :param position: 相机位置
    :param target: 目标位置（默认是原点）
    :param up: 定义相机的上方向（默认是 Y 轴方向）
    :return: 四元数 (x, y, z, w)
    """
    # 计算前向向量（从相机指向目标）
    forward = -(target - position)
    forward /= np.linalg.norm(forward)

    # 计算右向量和上向量
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up_corrected = np.cross(forward, right)

    # 旋转矩阵（列向量形式）
    rotation_matrix = np.array([right, up_corrected, -forward]).T

    # 将旋转矩阵转换为四元数
    return R.from_matrix(rotation_matrix).as_quat()

def rotation_matrix_from_axis_angle(axis, angle):
    """
    根据旋转轴和角度构造旋转矩阵。

    :param axis: 旋转轴 (x, y, z)，需要归一化
    :param angle: 旋转角度（弧度）
    :return: 旋转矩阵 (3x3)
    """
    axis = axis / np.linalg.norm(axis)  # 归一化旋转轴
    x, y, z = axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # 构造反对称矩阵 K
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    # 计算旋转矩阵
    rotation_matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
    return rotation_matrix

def generate_HEALPix_viewpoints(n_side=2, radius=2, original_viewpoint=np.array([0,0,1]), offset_phi=0):
    num_pixels = hp.nside2npix(n_side)  # 计算总像素数
    theta, phi = hp.pix2ang(n_side, np.arange(num_pixels))  # 获取每个像素的中心坐标 (θ, φ)

    # 转换为笛卡尔坐标
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.stack([x, y, z], axis=-1)
    points = rotate_to_target(points, np.array([0, 0, 1]))
    points = rotate_around_z(points, offset_phi)
    points = rotate_to_target(points, original_viewpoint)

    # **重置旋转到默认朝向**
    poses = xyz2pose(points[:, 0], points[:, 1], points[:, 2])
    default_quat = poses[0,:4]

    # **绕轴旋转**
    rotation_axis = -torch.tensor([1,0,0])
    cos_half_angle = torch.cos(torch.tensor(offset_phi)  / 2)
    sin_half_angle = torch.sin(torch.tensor(offset_phi) / 2)
    rotation_quat = torch.tensor([cos_half_angle, *(sin_half_angle * rotation_axis)])
    q_0 = quaternion_multiply(rotation_quat, poses[0,:4])
    # axis = camera_position / np.linalg.norm(camera_position)
    # R_matrix = rotation_matrix_from_axis_angle(axis, offset_phi)
    # if offset_phi != 0:
    #     rotation_quat = R.from_rotvec(offset_phi * np.array([0,0,1])).as_quat()
    #     # 将旋转复合到初始四元数上
    #     q_0 = (R.from_quat(rotation_quat)*R.from_quat(default_quat)).as_quat()
    # else:
    #     q_0 = default_quat

    # **验证方向**
    # print("Initial direction:", R.from_quat(default_quat).apply(np.array([0, 0, -1])))
    # print("Rotated direction:", R.from_quat(q_0).apply(np.array([0, 0, -1])))

    # 更新 pose
    poses[0, :4] = torch.tensor(q_0)

    return poses

def generate_fibonacci_viewpoints(num_points, radius=2, original_viewpoint=None):
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = 2 * np.pi * indices / ((1 + np.sqrt(5)) / 2)  # 黄金比例螺旋
    theta = np.arccos(1 - 2 * indices / num_points)

    # 球面坐标转换为笛卡尔坐标
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    points = np.stack([x, y, z], axis=-1)
    if original_viewpoint is None:
        original_viewpoint = np.array([0, 0, 1])
    points = rotate_to_target(points, original_viewpoint)

    poses = xyz2pose(points[:,0],points[:,1],points[:,2])

    return poses

def generate_polar_viewpoints(num_azimuth=8,num_elevation=8,radius=[2],original_viewpoint=None):
    azimuths = np.linspace(0, 2 * np.pi, num=num_azimuth, endpoint=False)
    elevations = np.linspace(0, np.pi, num=num_elevation)
    radius = np.asarray(radius)
    azimuths,elevations,radius = np.meshgrid(azimuths, elevations, radius)
    azimuths = azimuths.ravel()
    elevations = elevations.ravel()
    radius = radius.ravel()

    x = radius * np.sin(elevations) * np.cos(azimuths)
    y = radius * np.sin(elevations) * np.sin(azimuths)
    z = radius * np.cos(elevations)

    points = np.stack([x, y, z], axis=-1)
    if original_viewpoint is None:
        original_viewpoint = np.array([0, 0, 1])
    points = rotate_to_target(points, original_viewpoint)

    poses = xyz2pose(points[:,0],points[:,1],points[:,2])

    return poses


def index2pose_HEALPix(index,n_side=2,radius=2):
    poses = generate_HEALPix_viewpoints(n_side)
    return poses[index]

def pose_from_xyz(xyz,original_viewpoint=np.array([0,0,1]), offset_phi=0):
    # input xyz numpy  13 
    points = rotate_to_target(xyz, np.array([0, 0, 1]))
    points = rotate_around_z(points, offset_phi)
    points = rotate_to_target(points, original_viewpoint)

    # **重置旋转到默认朝向**
    poses = xyz2pose(points[:, 0], points[:, 1], points[:, 2])
    default_quat = poses[0,:4]

    # **绕轴旋转**
    rotation_axis = -torch.tensor([1,0,0])
    cos_half_angle = torch.cos(torch.tensor(offset_phi)  / 2)
    sin_half_angle = torch.sin(torch.tensor(offset_phi) / 2)
    rotation_quat = torch.tensor([cos_half_angle, *(sin_half_angle * rotation_axis)])
    q_0 = quaternion_multiply(rotation_quat, poses[0,:4])

    # 更新 pose
    poses[0, :4] = torch.tensor(q_0)

    return poses