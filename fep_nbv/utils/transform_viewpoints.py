import numpy as np
import torch
import mathutils
from scipy.spatial.transform import Rotation as R


def xyz2pose(x,y,z):
    # 输入xyz的list，输出xyzwxyz的
    cameras = np.stack([x, y, z], axis=1)
    cameras = torch.tensor(cameras, dtype=torch.float32)
    directions = -cameras # 生成相机的方向，从相机位置指向原点
    
    directions = [mathutils.Vector(row) for row in directions.cpu().numpy()]
    for direction in directions:
        direction = direction.normalize()
    rot_quat = torch.tensor([direction.to_track_quat('-Z', 'Y') for direction in directions])

    pose_hist = torch.concat((rot_quat[:,[1,2,3,0]], cameras),dim=1)
    return pose_hist

def polar2pose(azimuths,elevations,radius):
    # 输入在方位角、俯仰角的采样点数，以及半径的采样点
    # 输出采样结果，其中方位角和俯仰角均匀采样
    # 按照 xyzw xyz的顺序排列
    x = radius * np.sin(elevations) * np.cos(azimuths)
    y = radius * np.sin(elevations) * np.sin(azimuths)
    z = radius * np.cos(elevations)

    pose = xyz2pose(x,y,z)
    return pose


def pose2xyz(poses):
    return poses[:,4],poses[:,5],poses[:,6]

def pose2polar(poses):
    x,y,z = poses[:,4],poses[:,5],poses[:,6]
    # 计算 r, theta, phi
    r = np.sqrt(x**2 + y**2 + z**2)  # 半径（可忽略）
    theta = np.arccos(z / r)  # 俯仰角，范围 [0, pi]
    phi = np.arctan2(y, x)  # 方位角，范围 [-pi, pi]
    # 将 phi 范围调整到 [0, 2*pi]
    phi[phi < 0] += 2 * np.pi

    return phi, theta

def xyz2polar(x,y,z):
    # 计算方位角 (Azimuth) 范围 [0, 2π]
    azimuth = np.arctan2(y, x)  # 计算范围 [-π, π]
    azimuth = np.mod(azimuth, 2 * np.pi)  # 确保在 [0, 2π] 范围内

    # 计算俯仰角 (Elevation) 范围 [0, π]
    r = np.sqrt(x**2 + y**2 + z**2)  # 计算半径
    elevation = np.arccos(z / r)  # 计算范围 [0, π]

    return azimuth, elevation

def to_transform(pose):
    """Convert quaternion & translation vector into transformation matrix"""
    transform = quat_to_rot_transform_torch(pose)

    # zzq version
    r = R.from_quat(pose[:4].cpu())
    transpose = torch.eye(4)
    transpose[:3,:3] = torch.tensor(r.as_matrix()) # type: ignore
    transpose[:3,3] = pose[4:]


    return transform