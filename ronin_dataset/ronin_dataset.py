import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mrob
from tqdm import tqdm
import sys
sys.path.insert(0,'./external/ronin/source')
from external.ronin.source.data_utils import load_cached_sequences
from external.ronin.source.data_glob_speed import GlobSpeedSequence

from spline_dataset.spline_dataloader import Spline_2D_Dataset

import quaternion


class RoninDataset(Spline_2D_Dataset):
    def __init__(self,
                data_dir = './data/seen_subjects_test_set',
                take_log_num=-1,
                window = 100,
                step = 10,
                stride= 1000,
                sampling_rate=200,
                subseq_len = 1,
                mode = 'test',
                enable_noise = not True):
        self.data_dir = data_dir
        self.window = window
        self.step = step
        self.stride = stride
        self.sampling_rate = 200.0
        self.subseq_len = subseq_len
        self.mode = mode
        self.input_dim = 6
        self.pose_dim = 3
        self.enable_noise = enable_noise
        self.take_log_num = take_log_num

        self.subsequences = [] 

        self.logs = list(Path(self.data_dir).rglob("**/data.hdf5"))

        self.logs = [x.parent.stem for x in self.logs] # TODO use all logs

        if take_log_num > -1:
            self.logs = self.logs[:take_log_num]

        self.seq_type = GlobSpeedSequence

        # all_imu = [gyro, acc] <- for ronin dataset this is the order of channels
        all_imu, all_velocities, all_aux = load_cached_sequences(self.seq_type, self.data_dir, self.logs, './out/ronin_cache')

        print(f"Found {len(all_imu)} experiments")

        self.all_acc = [x[:,3:] for x in all_imu]
        self.all_gyro = [x[:,:3] for x in all_imu]
        self.all_velocities = all_velocities
        
        self.all_xyz = [aux[:,5:] for aux in all_aux]
        self.all_R = [quaternion.as_rotation_matrix(quaternion.from_float_array(aux[:,1:5])) for aux in all_aux]

        self.all_gt_poses = [None for _ in range(len(self.all_R))]
        for idx, (R,xyz) in tqdm(enumerate(zip(self.all_R,self.all_xyz))):
            assert len(R) == len(xyz)
            self.all_gt_poses[idx] = np.array([np.concatenate((tr[:2], mrob.SO3(r).Ln()[2:])) for r,tr in zip(R,xyz)])

        # computing all indexes for subsequences of slices
        for exp_id, name in enumerate(self.logs):
             for start_idx in range(0, len(self.all_acc[exp_id]) - self.step*(self.subseq_len - 1) - self.window, self.stride):
                slices_idxs = []
                for sub_idx in range(self.subseq_len):
                    end_idx = start_idx + sub_idx*self.step + self.window
                    slices_idxs.append((start_idx + self.step*sub_idx, end_idx))
                self.subsequences.append((exp_id, slices_idxs))        

   
if __name__ == "__main__":


    dataset = RoninDataset(take_log_num=1,step=20,stride=1000, subseq_len=1000,mode='train')
    print(len(dataset))


    for i in range(len(dataset)):
        print(dataset.__getitem__(i))