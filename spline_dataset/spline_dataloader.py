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


class Spline_2D_Dataset_old():
    def __init__(self, spline_path, window = 100, enable_noise = True):

        self.window = window

        self.bias_acc = np.array([0,0])
        if enable_noise:
            self.Q_acc = np.array([0.9**2,0,0,0.9**2]).reshape(2,2)
        else:
            self.Q_acc = np.zeros((2,2))

        self.bias_w = np.array([0])
        if enable_noise:
            self.Q_w = np.array([0.15**2]).reshape(1,1)
        else:
            self.Q_w = np.zeros((1,1))

        paths = glob.glob(f'{spline_path}/spline_*.txt')
        print(f"Found {len(paths)} splines in path: {spline_path}")

        B = len(paths) # this is our batch size

        self.data = np.zeros((B,window,3))#np.concatenate((acc,gyro),axis=1)

        self.slices = [[] for _ in range(B)]
        self.gt_odometry = [[] for _ in range(B)]
        self.gt_traj = [[] for _ in range(B)]
        self.gt_poses = [[] for _ in range(B)]
        self.gt_velocity = [[] for _ in range(B)]
        self.w_z = [[] for _ in range(B)]

        for b in tqdm(range(len(paths))):
            # 1 sample == 1 spline

            spline_points = np.genfromtxt(paths[b])

            self.gt_traj[b] = spline_points

            acc, gyro, velocity, poses, tau = generate_imu_data(spline_points)
            number_elements_to_cut = 10
            self.gt_traj[b] = self.gt_traj[b][number_elements_to_cut:-number_elements_to_cut]
            acc = acc[number_elements_to_cut:-number_elements_to_cut]
            gyro = gyro[number_elements_to_cut:-number_elements_to_cut]
            tau = tau[number_elements_to_cut:-number_elements_to_cut]
            velocity = velocity[number_elements_to_cut:-number_elements_to_cut]
            poses = poses[number_elements_to_cut:-number_elements_to_cut]
            #TODO cut begining and end of splines
            # TODO inject noise:
            # -additive
            # -multiplicative

            tmp_data = np.concatenate((acc,gyro),axis=1)
            
            # splitting 1 track into several slices
            slice_num = tmp_data.shape[0] // self.window
            for i in range(slice_num):

                temp_slice = tmp_data[i*self.window : (i+1)*self.window]

                acc_noise = np.random.multivariate_normal(self.bias_acc,self.Q_acc,self.window)
                temp_slice[:,:2] = temp_slice[:,:2] + acc_noise

                omega_noise = np.random.multivariate_normal(self.bias_w, self.Q_w,self.window)
                temp_slice[:,2:] = temp_slice[:,2:] + omega_noise

                self.slices[b].append(temp_slice) # adding i-th slice into b-th track

            # self.gt_poses[b] = np.concatenate((self.gt_traj[b][:tau.shape[0]], tau), axis=1)[::window]
            self.gt_poses[b] = poses[::window]
            self.gt_velocity[b] = velocity[::window]
            self.w_z[b] = gyro[::window]

        self.X = np.array(self.slices)
        #self.gt_traj = np.array(self.gt_traj)
        self.gt_poses = np.array(self.gt_poses)
        self.gt_velocity = np.array(self.gt_velocity)
        self.w_z = np.array(self.w_z)
        
        self.adjust_shape()
        
        self.gt_poses_se3 = convert_to_se3(self.gt_poses)
            
    def adjust_shape(self):
        min_length = min(self.X.shape[1], self.gt_velocity.shape[1])
        self.X = self.X[:, :min_length, :, :]
        self.X = self.X.reshape(-1, self.X.shape[2], self.X.shape[3])
        self.w_z = self.X[:, :, 2]
        
        self.gt_poses = self.gt_poses[:, :min_length, :]
        self.gt_poses = self.gt_poses.reshape(-1, self.gt_poses.shape[-1])
        
        self.gt_velocity = self.gt_velocity[:, :min_length, :]
        self.gt_velocity = self.gt_velocity.reshape(-1, self.gt_velocity.shape[-1])
        
        self.w_z = self.w_z[:min_length, :]
        self.w_z = self.w_z.reshape(-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        imu = torch.tensor(self.X[idx], dtype=torch.float32).permute(1, 0)
        velocity = torch.tensor(self.gt_velocity[idx], dtype=torch.float32)
        poses = torch.tensor(self.gt_poses[idx], dtype=torch.float32)
        poses_se3 = torch.tensor(self.gt_poses_se3[idx], dtype=torch.float32)
        w_z = torch.tensor(self.w_z[idx], dtype=torch.float32)
        return imu, velocity, poses, w_z


class Spline_2D_Dataset(Dataset):
    def __init__(self, spline_path, window=100, subseq_len=4, enable_noise=False):
        self.window = window
        self.subseq_len = subseq_len
        self.subsequences = []  # List of (experiment_idx, start_idx)

        self.bias_acc = np.array([0, 0])
        self.Q_acc = np.diag([0.9**2, 0.9**2]) if enable_noise else np.zeros((2, 2))
        self.bias_w = np.array([0])
        self.Q_w = np.array([[0.15**2]]) if enable_noise else np.zeros((1, 1))

        self.all_slices = []     # List of [num_slices, window, 3]
        self.all_velocities = [] # List of [num_slices, 2]
        self.all_gt_poses = []   # List of [num_slices, 3]

        paths = glob.glob(f'{spline_path}/spline_*.txt')
        print(f"Found {len(paths)} splines in path: {spline_path}")

        for exp_id, path in enumerate(tqdm(paths)):
            spline_points = np.genfromtxt(path)
            acc, gyro, velocity, poses, tau = generate_imu_data(spline_points)

            acc = acc[10:-10]
            gyro = gyro[10:-10]
            velocity = velocity[10:-10]
            poses = poses[10:-10]

            tmp_data = np.concatenate((acc, gyro), axis=1)
            num_slices = tmp_data.shape[0] // window

            if num_slices < subseq_len:
                continue  # Skip if not enough slices for a full subsequence

            slices = []
            vel_list = []
            pose_list = []

            for i in range(num_slices):
                imu_slice = tmp_data[i * window:(i + 1) * window]

                acc_noise = np.random.multivariate_normal(self.bias_acc, self.Q_acc, self.window)
                omega_noise = np.random.multivariate_normal(self.bias_w, self.Q_w, self.window)

                imu_slice[:, :2] += acc_noise
                imu_slice[:, 2:] += omega_noise

                slices.append(imu_slice)
                vel_list.append(velocity[i * window])
                pose_list.append(poses[i * window])

            self.all_slices.append(np.stack(slices))        # [num_slices, window, 3]
            self.all_velocities.append(np.stack(vel_list))  # [num_slices, 2]
            self.all_gt_poses.append(np.stack(pose_list))   # [num_slices, 3]

            for start_idx in range(0, num_slices - subseq_len + 1):
                self.subsequences.append((exp_id, start_idx))

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, idx):
        exp_id, start_idx = self.subsequences[idx]
        slices = self.all_slices[exp_id][start_idx:start_idx + self.subseq_len]       # [M, window, 3]
        velocities = self.all_velocities[exp_id][start_idx:start_idx + self.subseq_len]  # [M, 2]
        gt_poses = self.all_gt_poses[exp_id][start_idx:start_idx + self.subseq_len]   # [M, 3]

        imu_seq = torch.tensor(slices, dtype=torch.float32).permute(0, 2, 1)          # [M, 3, window]
        vel_seq = torch.tensor(velocities, dtype=torch.float32)                      # [M, 2]
        gt_pose_seq = torch.tensor(gt_poses, dtype=torch.float32)                    # [M, 3]

        return imu_seq, vel_seq, gt_pose_seq
    

if __name__ == "__main__":
    output_path = './out/'

    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    path_to_splines = output_path + 'splines/'

    number_of_splines = 10
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)

    if not os.path.isfile(path_to_splines + f'spline_dataset_{number_of_splines}.pkl'):
        dataset = Spline_2D_Dataset(path_to_splines, window=10, enable_noise = not True)
        pickle.dump(dataset,open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','wb'))
    else:
        dataset = pickle.load(open(path_to_splines + f'spline_dataset_{number_of_splines}.pkl','rb'))
    dataset = Spline_2D_Dataset(path_to_splines, window=10, enable_noise= not True)

    print(f"{dataset.__len__()=}")
    print(f"{dataset.__getitem__(0)=}")

    # for i in range(len(dataset)):
    #     #sample: dict_keys(['imu', 'y', 'gt_traj', 'gt_poses', 'gt_velocity', 'gt_orientation', 'gt_se3', 'gt_se3vel', 'time])

    #     s = dataset.__getitem__(i)

    #     plt.plot(s['time'],s['imu'][...,0],label='acc_x')
    #     plt.plot(s['time'],s['imu'][...,1],label='acc_y')
    #     plt.plot(s['time'],s['imu'][...,2],label='omega_z')
    #     plt.grid()
    #     plt.legend()


    #     plt.figure()
    #     plt.plot(s['gt_traj'][...,0],s['gt_traj'][...,1],label='gt traj')
    #     plt.grid()
    #     plt.axis('equal')
    #     plt.legend()

    #     plt.show()

    N = len(dataset)

    for i in range(N):
        print(dataset.__getitem__(i)[0].shape)
        print(dataset.__getitem__(i)[1].shape)