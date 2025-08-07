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
        slices of IMU data with shape [S,W,C], where S - length of subsequence,
        W - input tensor length, C - input tensor channels.
        Args:
            spline_path (_type_): path to spline files
            window (int, optional): Odometry model input size in time. Defaults to 100.
            step (int, optional): Step between two windows in subsequences. Defaults to 10.
            subseq_len (int, optional): Number of windows in subsequence. Defaults to 1.
            enable_noise (bool, optional): IMU noise injection. Defaults to False.
        """
    def __init__(self, spline_path, window = 100, step = 10, stride=10, sampling_rate=100, subseq_len = 1, enable_noise = False):
        self.spline_path = spline_path
        self.window = window
        self.input_dim = 3
        self.pose_dim = 3
        self.step = step
        self.sampling_rate = sampling_rate
        self.subseq_len = subseq_len
        # TODO add self.stride parameter - offset between two sequential subsequences. now this value is equal to step which is actualy independent from stride
        self.stride = stride
        self.enable_noise = enable_noise
        self.subsequences = []  # List of (experiment_idx, ((slice_1_range),(slice_2_range),...))

        if self.enable_noise:
            self.bias_acc = np.array([0, 0])
            self.Q_acc = np.diag([0.9**2, 0.9**2]) if enable_noise else np.zeros((2, 2))
            self.bias_w = np.array([0])
            self.Q_w = np.array([[0.15**2]]) if enable_noise else np.zeros((1, 1))

        self.all_velocities = []
        self.all_gt_poses = []

        self.all_acc = []
        self.all_gyro = []
        self.all_tau = []

        self.paths = glob.glob(f'{spline_path}/spline_*.txt')
        print(f"Found {len(self.paths)} splines in path: {spline_path}")

        # iterating over all files with spline points and computing subsequence indexes
        for exp_id, path in enumerate(self.paths):
            spline_points = np.genfromtxt(path)
            acc, gyro, velocity, poses, tau = generate_imu_data(spline_points)

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
            self.all_acc.append(acc) # [T, 2]
            self.all_gyro.append(gyro) # [T, 1]
            self.all_velocities.append(velocity)  # [T, 2]
            self.all_gt_poses.append(poses)   # [T, 3]
            # self.all_tau.append(tau) # [T, 2]

            # computing all indexes for subsequences of slices
            # S - number of sequential slices of length W = self.window
            # TODO add self stride here instead of step
            for start_idx in range(0, len(tmp_data) - self.step*(self.subseq_len - 1) - self.window, self.stride):
                slices_idxs = []
                for sub_idx in range(self.subseq_len):
                    end_idx = start_idx + sub_idx*self.step + self.window
                    slices_idxs.append((start_idx + self.step*sub_idx, end_idx))
                self.subsequences.append((exp_id, slices_idxs))


    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        # this function will return evey object with leading dimension S:
        # imu slices - [S, 3, W]
        # velocities - [S, 2]
        # gt_poses - [S, 3]
        exp_id, subseq_idxs = self.subsequences[idx]

        slices = [None for _ in range(self.subseq_len)]
        velocities = [None for _ in range(self.subseq_len)]
        gt_poses = [None for _ in range(self.subseq_len)]

        for local_idx, sub_idx in enumerate(subseq_idxs):
            subseq_range = range(sub_idx[0], sub_idx[1])
            slices[local_idx] = np.hstack((self.all_acc[exp_id][subseq_range],self.all_gyro[exp_id][subseq_range]))
            velocities[local_idx] = self.all_velocities[exp_id][subseq_range].mean(axis=0)
            gt_poses[local_idx] = self.all_gt_poses[exp_id][subseq_range][-1]

        slices = np.array(slices)
        velocities = np.array(velocities)
        gt_poses = np.array(gt_poses)

        imu_seq = torch.tensor(slices, dtype=torch.float32).permute(0, 2, 1)    # [S, 3, window]
        vel_seq = torch.tensor(velocities, dtype=torch.float32)                 # [S, 2]
        gt_pose_seq = torch.tensor(gt_poses, dtype=torch.float32)               # [S, 3]

        assert imu_seq.shape == (self.subseq_len, self.input_dim, self.window)
        assert vel_seq.shape == (self.subseq_len, 2)
        assert gt_pose_seq.shape == (self.subseq_len, self.pose_dim)

        # TODO enable augmentation - rotate IMU and GT by the same random angle around z axis

        # TODO add random noise if enabled
        if self.enable_noise:
            acc_noise = np.random.multivariate_normal(self.bias_acc, self.Q_acc, self.window)
            omega_noise = np.random.multivariate_normal(self.bias_w, self.Q_w, self.window)

            imu_slice[:, :2] += acc_noise
            imu_slice[:, 2:] += omega_noise

        sample = {
            'imu_seq' : imu_seq,
            'gt_vel_seq' : vel_seq,
            'gt_poses_seq' : gt_pose_seq,
            # 'gt_poses_se3' : np.array([mrob.SE3([g[2],0,0,g[0],g[1],0]) for g in gt_pose_seq]),
        }
        return sample
    
    def get_collate_fn(self):
        def regular_collate(batch):
            imu = torch.stack([item['imu_seq'] for item in batch])
            gt_vel = torch.stack([item['gt_vel_seq'] for item in batch])
            gt_poses = torch.stack([item['gt_poses_seq'] for item in batch])
            return {'imu': imu, 'gt_vel': gt_vel, 'gt_poses': gt_poses}
        return regular_collate



# TODO need to merge SplineIMUDataset class into Spline_2D_Dataset - the difference is small
# just need to add mode argument so either prepare and collate data for first, second or for both modes
# experiments `fgo_nn_splines` and `diffusion_splines` must work from the same dataset and data
class SplineIMUDataset(Dataset):
    def __init__(self, spline_path=None, num_samples=20, window_size=200, stride=20,
                 enable_noise=True, mode='both', noise_level=0.2, imu_freq=100.0):
        """
        Dataset for 2D spline trajectories with IMU data generation.
        
        Args:
            spline_path: Path to spline files (if None, generates synthetic splines)
            num_samples: Number of synthetic samples to generate (if spline_path is None)
            window_size: Size of sliding window for regular models
            stride: Stride for sliding window
            enable_noise: Whether to add noise to measurements
            mode: 'odometry' for supervised odometry learning or 'diffusion' for diffusion models or 'both' for both
            noise_level: Amount of noise for diffusion models (0-1)
            imu_freq: IMU sampling frequency in Hz
        """
        self.samples = []
        self.window_size = window_size
        self.stride = stride
        self.enable_noise = enable_noise
        self.mode = mode

        assert self.mode in ['odometry', 'diffusion', 'both']

        self.noise_level = noise_level
        self.dt = 1.0 / imu_freq
        
        # Noise parameters
        self.bias_acc = np.array([0.1, 0.1])
        self.bias_w = np.array([0.1])
        self.Q_acc = np.diag([0.8**2, 0.8**2]) if enable_noise else np.zeros((2, 2))
        self.Q_w = np.diag([0.15**2]) if enable_noise else np.zeros((1, 1))
        
        if spline_path is not None:
            self._load_splines_from_files(spline_path)
        else:
            self._generate_synthetic_splines(num_samples)
    
    def _load_splines_from_files(self, spline_path):
        """Load splines from text files and generate IMU data."""
        paths = glob.glob(f'{spline_path}/spline_*.txt')
        print(f"Found {len(paths)} spline files.")
        
        for path in tqdm(paths):
            spline_points = np.genfromtxt(path)
            self._process_spline(spline_points)
    
    def _generate_synthetic_splines(self, num_samples):
        """Generate synthetic spline trajectories."""
        for _ in tqdm(range(num_samples)):
            # Generate random control points
            num_ctrl_points = np.random.randint(3, 7)
            ctrl_points = np.random.uniform(-1, 1, (num_ctrl_points, 2))
            
            # Create cubic spline
            t = np.linspace(0, 1, num_ctrl_points)
            cs_x = CubicSpline(t, ctrl_points[:, 0])
            cs_y = CubicSpline(t, ctrl_points[:, 1])
            
            # Sample spline at regular intervals
            t_samples = np.linspace(0, 1, 200)  # Fixed number of points
            spline_points = np.column_stack((cs_x(t_samples), cs_y(t_samples)))
            
            self._process_spline(spline_points)
    
    def _process_spline(self, spline_points):
        """Generate IMU data from spline points and create samples."""
        acc, gyro, velocity, poses, tau = generate_imu_data(spline_points)
        w_z = gyro
        
        if self.enable_noise:
            noisy_acc = acc + np.random.multivariate_normal(self.bias_acc, self.Q_acc, acc.shape[0])
            noisy_gyro = gyro + np.random.multivariate_normal(self.bias_w, self.Q_w, gyro.shape[0]).reshape(-1, 1)
        else:
            noisy_acc = acc.copy()
            noisy_gyro = gyro.copy()
        
        imu_clean = np.concatenate([acc, gyro], axis=1)  # [T, 3]
        imu_noisy = np.concatenate([noisy_acc, noisy_gyro], axis=1)  # [T, 3]
        T = imu_clean.shape[0]
        
        for i in range(0, T - self.window_size + 1, self.stride):
            sample = {}
            if self.mode in ['odometry', 'both']:
                # For regular odometry models, store windowed IMU with ground truth
                imu_win = imu_noisy[i:i + self.window_size].T  # [3, W]
                vel_win = velocity[i:i + self.window_size]  # [W, 2]
                pose_win = poses[i:i + self.window_size]  # [W, 3]
                w_z_win = w_z[i:i + self.window_size]     # [W, 1]
                
                sample.update({
                    'imu': torch.tensor(imu_win, dtype=torch.float32),            # [3, W]
                    'gt_vel': torch.tensor(vel_win.mean(axis=0), dtype=torch.float32),  # [2]
                    'gt_poses': torch.tensor(pose_win.mean(axis=0), dtype=torch.float32),# [3]
                    'w_z': torch.tensor(w_z_win.mean(), dtype=torch.float32).unsqueeze(0)  # [1]
                })

            if self.mode in ['diffusion', 'both']:
                # For diffusion models, store pairs of clean and noisy IMU data
                clean_win = imu_clean[i:i + self.window_size].T  # [3, W]
                
                sample.update({
                    'clean_imu': torch.tensor(clean_win, dtype=torch.float32),
                })
            
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_collate_fn(self):        
        def diffusion_collate(batch):
            noisy = torch.stack([item['imu'] for item in batch])
            clean = torch.stack([item['clean_imu'] for item in batch])
        
            imu = torch.stack([item['imu'] for item in batch])
            gt_vel = torch.stack([item['gt_vel'] for item in batch])
            gt_poses = torch.stack([item['gt_poses'] for item in batch])
            w_z = torch.stack([item['w_z'] for item in batch])
            return {'imu': imu, 'gt_vel': gt_vel, 'gt_poses': gt_poses, 'w_z': w_z, 'noisy': noisy, 'clean': clean}
        return diffusion_collate


if __name__ == "__main__":
    output_path = './out/'

    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    path_to_splines = output_path + 'splines/'

    number_of_splines = 10
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)

    dataset = Spline_2D_Dataset(path_to_splines, window=100,subseq_len=5, enable_noise= not True)

    print(f"{dataset.__len__()=}")
    print(f"{dataset.__getitem__(0)=}")

    N = len(dataset)

    for i in range(N):
        print(dataset.__getitem__(i))