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

def add_noise_to_poses(poses, pos_std=4.0):
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
        x, y, theta = pose

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
    def __init__(self, spline_path, window_size=20, stride=1, enable_noise=True):
        self.samples = []
        self.window_size = window_size

        self.bias_acc = np.array([0.1, 0.1])
        self.bias_w = np.array([0.1])
        self.Q_acc = np.diag([0.8**2, 0.8**2]) if enable_noise else np.zeros((2, 2))
        self.Q_w = np.diag([0.15**2]) if enable_noise else np.zeros((1, 1))

        paths = glob.glob(f'{spline_path}/spline_*.txt')
        print(f"Found {len(paths)} spline files.")

        samples = []
        for path in tqdm(paths):
            spline_points = np.genfromtxt(path)
            acc, gyro, velocity, poses = generate_imu_data(spline_points)
            w_z = gyro

            if enable_noise:
                acc += np.random.multivariate_normal(self.bias_acc, self.Q_acc, acc.shape[0])
                gyro += np.random.multivariate_normal(self.bias_w, self.Q_w, gyro.shape[0]).reshape(-1, 1)

            imu = np.concatenate([acc, gyro], axis=1)  # [T, 3]
            T = imu.shape[0]

            for i in range(0, T - window_size + 1, stride):
                imu_win = imu[i:i + window_size].T  # [3, window]
                vel_win = velocity[i:i + window_size]  # [window, 2]
                pose_win = poses[i:i + window_size]  # [window, 3]
                w_z_win = w_z[i:i + window_size]     # [window, 1]

                sample = {
                    'imu': torch.tensor(imu_win, dtype=torch.float32),            # [3, W]
                    'gt_vel': torch.tensor(vel_win.mean(axis=0), dtype=torch.float32),  # [2]
                    'gt_poses': torch.tensor(pose_win.mean(axis=0), dtype=torch.float32),# [3]
                    'w_z': torch.tensor(w_z_win.mean(), dtype=torch.float32).unsqueeze(0)  # [1]
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    path_to_splines = "./out/spline_dataset"
    number_of_splines = 10
    
    # if not os.path.exists(path_to_splines):
    number_of_control_nodes = 10
    generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)
    
    dataset = Spline_2D_Dataset(path_to_splines)
    print(len(dataset))
    # noisy_pose, gt_pose= dataset[0]
    # print(noisy_pose.shape, gt_pose.shape)