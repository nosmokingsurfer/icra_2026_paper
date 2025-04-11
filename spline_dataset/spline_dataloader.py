import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
import mrob
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import torch
from torch.utils.data import Dataset

from spline_dataset.spline_diff import generate_imu_data
from spline_dataset.spline_generation import generate_batch_of_splines

def add_noise_to_poses(poses, pos_std=0.9, angle_std=0.001):
    poses = poses.clone() if isinstance(poses, torch.Tensor) else poses.copy()

    x = poses[:, 0]
    y = poses[:, 1]

    x_noisy = x + np.random.normal(0, pos_std, size=x.shape)
    y_noisy = y + np.random.normal(0, pos_std, size=y.shape)

    noisy_poses = np.stack([x_noisy, y_noisy, poses[:, 2]], axis=1)
    return noisy_poses
    
    
def convert_to_se3(poses):
    poses = np.array(poses) 
    single_input = False
    if poses.ndim == 1 and poses.shape[0] == 3:
        poses = poses[None, :]  # make it shape (1, 3)
        single_input = True

    num_poses = poses.shape[0]
    se3_matrices = np.zeros((num_poses, 4, 4))

    for i, pose in enumerate(poses):
        x, y, theta= pose

        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, 0]
        se3_matrices[i] = T

    if single_input:
        return se3_matrices[0]
    return se3_matrices


class Spline_2D_Dataset(Dataset):
    def __init__(self, spline_path, enable_noise=False):
        self.imu_data = []
        self.gt_velocity = []
        self.gt_poses = []
        self.gt_poses_se3 = []

        self.bias_acc = np.array([0, 0])
        self.bias_w = np.array([0])
        
        self.Q_acc = np.diag([0.9**2, 0.9**2]) if enable_noise else np.zeros((2, 2))
        self.Q_w = np.diag([0.15**2]) if enable_noise else np.zeros((1, 1))

        paths = glob.glob(f'{spline_path}/spline_*.txt')
        print(f"Found {len(paths)} spline files.")

        for path in tqdm(paths):
            spline_points = np.genfromtxt(path)
            acc, gyro, tau, _, velocity, _ = generate_imu_data(spline_points)

            if enable_noise:
                acc += np.random.multivariate_normal(self.bias_acc, self.Q_acc, acc.shape[0])
                gyro += np.random.multivariate_normal(self.bias_w, self.Q_w, gyro.shape[0]).reshape(-1, 1)

            imu = np.concatenate([acc, gyro], axis=1)
            yaw = np.arctan2(tau[:, 1], tau[:, 0])
            poses = np.concatenate([spline_points[:20], yaw[:20, None]], axis=1) # spline_points[:tau.shape[0]], tau] len(yaw)
            se3 = convert_to_se3(poses)

            self.imu_data.append(torch.tensor(imu, dtype=torch.float32).T)  # shape: [3, T]
            self.gt_velocity.append(torch.tensor(velocity, dtype=torch.float32))  # [T, 2]
            self.gt_poses.append(torch.tensor(poses, dtype=torch.float32))  # [T, 4]
            self.gt_poses_se3.append(torch.tensor(se3, dtype=torch.float32))  # [T, 4, 4]

    def __len__(self):
        return len(self.imu_data)

    def __getitem__(self, idx):
        imu = self.imu_data[idx]            # [3, T], in **body frame**
        velocity = self.gt_velocity[idx]    # [T, 2], in **world frame**
        poses = self.gt_poses[idx]          # [T, 4], in **world frame** (x, y, cos, sin)
        poses_se3 = self.gt_poses_se3[idx]  # [T, 4, 4], SE(3) world poses
        
        x = torch.tensor(add_noise_to_poses(poses), dtype=torch.float32)
        y = poses.clone().float()

        return x, y


if __name__ == "__main__":
    path_to_splines = "./out/spline_dataset"
    number_of_splines = 10
    
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)
    
    dataset = Spline_2D_Dataset(path_to_splines, enable_noise=False)
    print(len(dataset))
    noisy_pose, gt_pose= dataset[0]
    print(noisy_pose.shape, gt_pose.shape)