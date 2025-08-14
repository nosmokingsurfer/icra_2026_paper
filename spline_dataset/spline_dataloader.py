# from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

from spline_dataset.spline_generation import generate_batch_of_splines

from spline_dataset.spline_diff import generate_imu_data
# from spline_generation import generate_batch_of_splines

import glob
from tqdm import tqdm
import mrob
import os
import pickle

import torch
from torch.utils.data import Dataset

def get_gt_se3_Poses(poses):
    result = [None]*len(poses)
    angles = np.zeros((len(poses),3))
    angles[:,2] = np.arctan2(poses[:,3],poses[:,2])
    xyz = np.zeros((len(poses),3))
    xyz[:,:2] = poses[:,:2]

    for i in range(len(poses)):
        result[i] = mrob.SE3(mrob.SO3(angles[i]),xyz[i])

    return result


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


def get_gt_se3vel_Poses(poses, velocities):
    result = [None]*len(poses)
    angles = np.zeros((len(poses),3))
    angles[:,2] = np.arctan2(poses[:,3],poses[:,2])
    xyz = np.zeros((len(poses),3))
    xyz[:,:2] = poses[:,:2]

    v_xyz = np.zeros((len(poses),3))
    v_xyz[:,:2] = velocities

    for i in range(len(poses)):
        result[i] = mrob.SE3vel(mrob.SO3(angles[i]),xyz[i], v_xyz[i])
    return result


class Spline_2D_Dataset(Dataset):
    """
        Reads synthetic generated spline data and prepares subsequences of overlapping 
        slices of IMU data with shape [S,C,W], where S - length of subsequence,
        W - input tensor length, C - input tensor channels.
        Args:
            spline_path (_type_): path to spline files
            window (int, optional): Odometry model input size in time. Defaults to 100.
            step (int, optional): Step between two windows in subsequences. Defaults to 10.
            stride (int, optional): Stride between two subsequences from same experiment. Default is 10.
            subseq_len (int, optional): Number of windows in subsequence. Defaults to 1.
            enable_noise (bool, optional): IMU noise injection. Defaults to False.
            mode (string, optional): Prepare subsequences for "regression", denoising "diffusion" or "both". Default is regression.
        """
    def __init__(self, spline_path, window = 100, step = 10, stride=10, sampling_rate=100, subseq_len = 1, mode='regression', enable_noise = False, noise_level = 0.2):
        self.spline_path = spline_path
        self.window = window
        self.input_dim = 3
        self.velocity_dim = 2
        self.pose_dim = 3
        self.step = step
        self.sampling_rate = sampling_rate
        self.subseq_len = subseq_len
        self.mode = mode
        self.stride = stride
        self.enable_noise = enable_noise
        self.noise_level = noise_level #TODO not used yet
        self.subsequences = []  # List of (experiment_idx, ((slice_1_range),(slice_2_range),...))

        assert self.mode in ['regression', 'diffusion', 'both']

        if self.enable_noise:
            self.bias_acc = np.array([0, 0])
            self.Q_acc = np.diag([0.9**2, 0.9**2]) if enable_noise else np.zeros((2, 2))
            self.bias_w = np.array([0])
            self.Q_w = np.array([[0.15**2]]) if enable_noise else np.zeros((1, 1))

        
        self.all_acc = []
        self.all_noisy_acc = []
        self.all_noisy_gyro = []
        self.all_gyro = []
        self.all_velocities = []
        self.all_gt_poses = []

        self.paths = glob.glob(f'{spline_path}/spline_*.txt')
        print(f"Found {len(self.paths)} splines in path: {spline_path}")

        # iterating over all files with spline points and computing subsequence indexes
        for exp_id, path in enumerate(self.paths):
            # Readin spline points from files and generating clean IMU data
            spline_points = np.genfromtxt(path)
            acc, gyro, velocity, poses, _ = generate_imu_data(spline_points, self.sampling_rate)

            # TODO trimming away heads and tails of spline IMU data becase it can be too big
            # acc = acc[10:-10]
            # gyro = gyro[10:-10]
            # velocity = velocity[10:-10]
            # poses = poses[10:-10]

            tmp_data = np.concatenate((acc, gyro), axis=1)

            num_slices = (tmp_data.shape[0] - self.window) // self.step

            if num_slices < self.subseq_len:
                continue  # Skip if not enough slices for at least one subsequence

            # storing all data as it is for every experiment so later can get values by indexes
            # T - experiment duration
            self.all_acc.append(acc) # [T, 2] (a_x, a_y)
            self.all_gyro.append(gyro) # [T, 1] (w_z)
            self.all_velocities.append(velocity)  # [T, 2] (v_x, v_y)
            self.all_gt_poses.append(poses)   # [T, 3] (x, y, theta)

            if self.enable_noise:
                # storing noised IMU data
                self.all_noisy_acc.append(acc + np.random.multivariate_normal(self.bias_acc, self.Q_acc, acc.shape[0]))
                self.all_noisy_gyro.append(gyro + np.random.multivariate_normal(self.bias_w, self.Q_w, gyro.shape[0]).reshape(-1, 1))
            else:
                # or just copy original signal
                self.all_noisy_acc.append(acc.copy())
                self.all_noisy_gyro.append(gyro.copy())

            # computing all indexes for subsequences of slices
            # S - number of sequential slices of length W = self.window
            for start_idx in range(0, len(tmp_data) - self.step*(self.subseq_len - 1) - self.window, self.stride):
                slices_idxs = []
                for sub_idx in range(self.subseq_len):
                    end_idx = start_idx + sub_idx*self.step + self.window
                    slices_idxs.append((start_idx + self.step*sub_idx, end_idx))
                self.subsequences.append((exp_id, slices_idxs))


    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        # this function will return every object with leading dimension S:
        # clean imu slices - [S, 3, W]
        # noisy imu slices - [S, 3, W]
        # velocities - [S, 2]
        # gt_poses - [S, 3]
        exp_id, subseq_idxs = self.subsequences[idx]

        clean_imu_seq = np.zeros((self.subseq_len, self.window, self.input_dim))
        noisy_imu_seq = np.zeros((self.subseq_len, self.window, self.input_dim))
        gt_vel_seq = np.zeros((self.subseq_len, self.velocity_dim))
        gt_poses_seq = np.zeros((self.subseq_len, self.pose_dim))

        for local_idx, sub_idx in enumerate(subseq_idxs):
            subseq_range = range(sub_idx[0], sub_idx[1])

            clean_imu_seq[local_idx] = np.hstack((self.all_acc[exp_id][subseq_range],self.all_gyro[exp_id][subseq_range]))
            noisy_imu_seq[local_idx] = np.hstack((self.all_noisy_acc[exp_id][subseq_range],self.all_noisy_gyro[exp_id][subseq_range]))
            gt_vel_seq[local_idx] = self.all_velocities[exp_id][subseq_range].mean(axis=0)
            gt_poses_seq[local_idx] = self.all_gt_poses[exp_id][subseq_range][-1]

        clean_imu_seq = torch.tensor(clean_imu_seq, dtype=torch.float32).permute(0, 2, 1)    # [S, 3, window]
        noisy_imu_seq = torch.tensor(noisy_imu_seq, dtype=torch.float32).permute(0, 2, 1)    # [S, 3, window]
        gt_vel_seq = torch.tensor(gt_vel_seq, dtype=torch.float32)                        # [S, 2]
        gt_poses_seq = torch.tensor(gt_poses_seq, dtype=torch.float32)               # [S, 3]

        assert clean_imu_seq.shape == (self.subseq_len, self.input_dim, self.window)
        assert noisy_imu_seq.shape == (self.subseq_len, self.input_dim, self.window)
        assert gt_vel_seq.shape == (self.subseq_len, 2)
        assert gt_poses_seq.shape == (self.subseq_len, self.pose_dim)

        # TODO enable augmentation - rotate IMU and GT by the same random angle around z axis

        sample = {
            'clean_imu' : clean_imu_seq,
            'noisy_imu' : noisy_imu_seq,
            'gt_vel' : gt_vel_seq,
            'gt_poses' : gt_poses_seq,
        }
        return sample
    
    def get_collate_fn(self):
        if self.mode == 'diffusion':
            def diffusion_collate(batch):
                clean_imu = torch.stack([item['clean_imu'] for item in batch])
                noisy_imu = torch.stack([item['noisy_imu'] for item in batch])
                return {'clean_imu': clean_imu, 'noisy_imu': noisy_imu}
            return diffusion_collate
        elif self.mode == 'regression':
            def regression_collate(batch):
                noisy_imu = torch.stack([item['noisy_imu'] for item in batch])
                gt_vel = torch.stack([item['gt_vel'] for item in batch])
                gt_poses = torch.stack([item['gt_poses'] for item in batch])
                return {'noisy_imu': noisy_imu, 'gt_vel': gt_vel, 'gt_poses': gt_poses}
            return regression_collate
        elif self.mode == 'both':
            def both_collate(batch):
                clean_imu = torch.stack([item['clean_imu'] for item in batch])
                noisy_imu = torch.stack([item['noisy_imu'] for item in batch])
                gt_vel = torch.stack([item['gt_vel'] for item in batch])
                gt_poses = torch.stack([item['gt_poses'] for item in batch])
                return {'clean_imu': clean_imu, 'noisy_imu': noisy_imu, 'gt_vel': gt_vel, 'gt_poses': gt_poses}
            return both_collate

if __name__ == "__main__":
    output_path = './out/'

    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    path_to_splines = output_path + 'splines/'

    number_of_splines = 10
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)

    dataset = Spline_2D_Dataset(path_to_splines, window=100, subseq_len=5, enable_noise= not True)

    print(f"{dataset.__len__()=}")
    print(f"{dataset.__getitem__(0)=}")

    N = len(dataset)

    for i in range(N):
        print(dataset.__getitem__(i))