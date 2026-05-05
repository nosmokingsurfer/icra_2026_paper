from tqdm import tqdm
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import quaternion

from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
import pymap3d as pm
import ahrs

import sys
sys.path.insert(0,'.')

from gsdc_dataset.utils import get_tasks_for_dataset


def zero_padding_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collated = {}
    
    keys = batch[0].keys()
    keys = [k for k in keys if k not in ["task"]]
    
    lengths = [item['acc'].shape[0] for item in batch]
    for key in keys:
        tensors = [item[key] for item in batch]
        max_len = max(t.shape[0] for t in tensors)
        
        if tensors[0].dim() == 2:
            feature_dim = tensors[0].shape[1]
            padded_tensors = []
            for t in tensors:
                padded = torch.zeros(max_len, feature_dim, dtype=t.dtype)
                padded[:t.shape[0], :] = t
                padded_tensors.append(padded)
        elif tensors[0].dim() == 1:
            padded_tensors = []
            for t in tensors:
                padded = torch.zeros(max_len, dtype=t.dtype)
                padded[:t.shape[0]] = t
                padded_tensors.append(padded)
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensors[0].dim()}")

        collated[key] = torch.stack(padded_tensors)
    
    collated['lengths'] = torch.tensor(lengths,dtype=torch.int64,device=batch[0]['acc'].device)
    collated['task'] = [t['task'] for t in batch]

    return collated

class GSDC_dataset(Dataset):
    def __init__(self, mode, tasks, data_path,  **args):
        # data_path = "./data/smartphone-decimeter-2022/"
        self.dataloader_params = args
        assert "window" in self.dataloader_params
        assert mode in ['train','val','test']

        self.data_path = data_path
        self.mode = mode

        self.tasks = tasks
        if self.mode == "train":
            pass
            #TODO set augmentations here yaw_transform


        # TODO apply white lists
        
        print(f"Processing {len(self.tasks)} tasks ...")
        self.global_index_accumulator = np.int64(0)

        self.task_indexes = [None] * len(self.tasks)
        self.task_data = {}

        for idx, task in tqdm(enumerate(self.tasks), total=len(self.tasks)):
            rotated_combined_data_path = './out_vdr/'+ f"{task['mode']}_{self.dataloader_params['frequency']}/" + task['sample_id'] + "_rotated_combined_data.pickle"
            tmp = pd.read_pickle(rotated_combined_data_path)
            task['length'] = len(tmp)
            number_of_steps = np.int64(np.floor(len(tmp)- self.dataloader_params['window'])/self.dataloader_params['step'])
            self.global_index_accumulator += number_of_steps
            self.task_indexes[idx] = [self.global_index_accumulator, task['sample_id'],number_of_steps]
            self.task_data[task['sample_id']] = tmp


    def __len__(self):
        if self.mode in ['train','val']:
            return self.global_index_accumulator
        elif self.mode == 'test':
            return len(self.task_indexes)

    def get_test_item(self, idx):
        global_index, sample_id, num_steps = self.task_indexes[idx]

        data = self.task_data[sample_id]

        quats_wi = data[['q_wi_w','q_wi_x','q_wi_y','q_wi_z']].values
        euler_wi = quaternion.as_euler_angles(quaternion.from_float_array(quats_wi))

        result = {
            "acc" : torch.tensor(data[['a_s_x','a_s_y','a_s_z']].values, dtype=torch.float32),
            "gyro" : torch.tensor(data[['w_s_x','w_s_y','w_s_z']].values, dtype=torch.float32),
            "gt_velocity" : torch.tensor(data[['gt_vel_x','gt_vel_y']].values, dtype=torch.float32),
            "gt_traj" : torch.tensor(data[['e','n']].values, dtype=torch.float32),
            "yaw_angle" : torch.tensor(euler_wi[:,0], dtype=torch.float32),
            "task" : self.tasks[idx]
        }

        # if len(result['acc'].shape) == 2:
        #     for k,v in result.items():
        #         result[k] = v.unsqueeze(0)

        return result


    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.get_test_item(idx)

        window = self.dataloader_params['window']
        step = self.dataloader_params['step']

        task_idx = np.searchsorted(np.array(self.task_indexes)[:,0].astype(np.int64),idx,side='right')
        global_idx, sample_id, num_steps = self.task_indexes[task_idx]

        local_slice_idx = np.int64(idx - (global_idx - num_steps))
        
        start_idx = np.int64(local_slice_idx*step)
        end_idx = np.int64(start_idx + window)

        data = self.task_data[sample_id].iloc[start_idx:end_idx]

        quats_wi = data[['q_wi_w','q_wi_x','q_wi_y','q_wi_z']].values
        euler_wi = quaternion.as_euler_angles(quaternion.from_float_array(quats_wi))

        result = {
            "acc" : torch.tensor(data[['a_s_x','a_s_y','a_s_z']].values, dtype=torch.float32),
            "gyro" : torch.tensor(data[['w_s_x','w_s_y','w_s_z']].values, dtype=torch.float32),
            "gt_velocity" : torch.tensor(data[['gt_vel_x','gt_vel_y']].values, dtype=torch.float32),
            "gt_traj" : torch.tensor(data[['e','n']].values, dtype=torch.float32),
            "yaw_angle" : torch.tensor(euler_wi[:,0], dtype=torch.float32)
        }

        # if len(result['acc'].shape) == 2:
        #     for k,v in result.items():
        #         result[k] = v.unsqueeze(0)


        # if (self.mode == "train"):
        #     prob = np.random.uniform(low=0,high=1)
        #     if prob < 0.7:
        #         random_rotation_degree = np.random.uniform(low=-180, high=180)
        #         random_rotation_radian = random_rotation_degree*2*np.pi/360.0
        #         random_rotation_matrix = quaternion.as_rotation_matrix(quaternion.from_euler_angles([random_rotation_radian,0,0]))

        #         random_rotation_matrix = torch.from_numpy(random_rotation_matrix).to(dtype=torch.float32)
        #         result["acc"] = acc_s = result["acc"] @ random_rotation_matrix
        #         result["gyro"] = result["gyro"] @ random_rotation_matrix
        #         result["gt_velocity"] = result["gt_velocity"] @ random_rotation_matrix[:2,:2]
            # apply augmentatnions here

        return result

def generate_combined_data(task, frequency=100):

    combined_data_path = './out_vdr/' + f"{task['mode']}_{frequency}/"+ task['sample_id'] + "_combined_data.pickle"
    if os.path.exists(combined_data_path):
        return

    if not os.path.exists('./out_vdr/'):
        os.makedirs('./out_vdr',exist_ok=True)
    
    if not os.path.exists(f"./out_vdr/{task['mode']}_{frequency}"):
        os.makedirs(f"./out_vdr/{task['mode']}_{frequency}",exist_ok=True)

    df = pd.read_csv(task['imu_file'])
    
    acc = df[df['MessageType'] == "UncalAccel"]
    acc = acc[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ']]
    acc.columns = ['t','a_x','a_y','a_z']
    acc['t'] = acc['t'].values/1e+3

    gyro = df[df['MessageType'] == "UncalGyro"]
    gyro = gyro[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ','BiasX','BiasY','BiasZ']]
    gyro.columns = ['t','w_x','w_y','w_z','bias_x','bias_y','bias_z']
    gyro['t'] = gyro['t'].values/1e+3

    # TODO add magnetometer
    # read uncalibrated messages

    gt_df = pd.read_csv(task['gt_file'])
    gt_df = gt_df[['UnixTimeMillis','LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters', 'SpeedMps', 'BearingDegrees']]
    gt_df.columns = ['t','lat','lon','alt','speed','bearing']
    gt_df['bearing'] = -1.*gt_df['bearing']+90
    gt_df['bearing'] = gt_df['bearing']*np.pi/180.
    gt_df['t'] = gt_df['t'].values/1e+3

    t_min = max(min(acc.t), min(gyro.t), min(gt_df.t))
    t_max = min(max(acc.t), max(gyro.t), max(gt_df.t))

    dt_step = 1./frequency
    
    t_new = np.arange(t_min, t_max, dt_step)

    acc_resampled = np.zeros((len(t_new),acc.shape[1]))
    acc_resampled[:,0] = t_new

    for i in range(acc.shape[1]-1):
        acc_resampled[:,i+1] = np.interp(t_new, acc['t'], acc.values[:,i+1])

    acc_resampled = pd.DataFrame(acc_resampled,columns =acc.columns)

    gyro_resampled = np.zeros((len(t_new), gyro.shape[1]))
    gyro_resampled[:,0] = t_new
    for i in range(gyro.shape[1]-1):
        gyro_resampled[:,i+1] = np.interp(t_new,gyro['t'], gyro.values[:,i+1])

    gyro_resampled = pd.DataFrame(gyro_resampled,columns =gyro.columns)
    
    gt_resampled = np.zeros((len(t_new), gt_df.shape[1]))
    gt_resampled[:,0] = t_new

    for i in range(gt_df.shape[1]-1):
        gt_resampled[:,i+1] = np.interp(t_new, gt_df['t'], gt_df.values[:,i+1])

    gt_resampled = pd.DataFrame(gt_resampled, columns = gt_df.columns)

    combined_data = pd.concat((acc_resampled, gyro_resampled, gt_resampled),axis=1)

    if not os.path.exists(combined_data_path):
        combined_data.to_pickle(combined_data_path)


def rotate_data(task, frequency=100):
    visualize = True
    sample_id = task['sample_id']

    rotated_combined_data_path = './out_vdr/'+ f"{task['mode']}_{frequency}/" + task['sample_id'] + "_rotated_combined_data.pickle"
    if os.path.exists(rotated_combined_data_path):
        return
    
    combined_data_path = './out_vdr/'+ f"{task['mode']}_{frequency}/" + task['sample_id'] + "_combined_data.pickle"

    combined_data = pd.read_pickle(combined_data_path)

    # step 1:
    # computing local ENU coordinates
    llh = combined_data[['lat','lon','alt']].values[0]
    llh[2] = 0 # TODO strong assumption
    lat = combined_data['lat'].values
    lon = combined_data['lon'].values
    alt = combined_data['alt'].values
    alt = 0 # TODO strong assumption

    e,n,u = pm.geodetic2enu(lat,lon,alt, llh[0],llh[1],llh[2])
    combined_data['e'] = e
    combined_data['n'] = n

    if visualize:
        plt.plot(e,n,label='gt trajectory')
        plt.title(f"{task['sample_id']}")
        plt.grid();plt.axis('equal');plt.legend()
        plt.savefig(f'./out_vdr/{task["mode"]}_{frequency}/{task["sample_id"]}_gt_trajectory.png')
        plt.close('all')


    # step 2:
    # computing attitude of IMU in local fixed frame:
    quats_wi = ahrs.filters.Madgwick(
        gyr= combined_data[['w_x','w_y','w_z']].values,
        acc = combined_data[['a_x','a_y','a_z']].values,
        frequency = frequency).Q

    quats_wi = quaternion.from_float_array(quats_wi)

    # yaw pitch roll from IMU to some fixed frame
    euler_wi = np.unwrap(quaternion.as_euler_angles(quats_wi),axis=0)

    if visualize:
        # Array with the 4 elements of quaternion of the form [w, x, y, z]
        plt.plot(np.unwrap(quaternion.as_euler_angles(quats_wi),axis=0),label=['yaw','pitch','roll'])
        plt.title('Euler angles from IMU frame to local fixed frame')
        plt.grid()

        plt.plot(np.unwrap(combined_data.bearing.values),'-',color='red', label='bearing from pvt')
        plt.legend()
        plt.savefig(f'./out_vdr/{task["mode"]}_{frequency}/{sample_id}_euler_w_to_imu.png')
        plt.close('all')

    # keeping only pitch and roll angles to rotate IMU frame to S-frame
    euler_si = euler_wi.copy()
    euler_si[:,0] = 0 # not changing the yaw angle when rotating
    quats_si = quaternion.from_euler_angles(euler_si)

    if visualize:
        R_si = quaternion.as_rotation_matrix(quaternion.from_euler_angles(euler_si))

        plt.plot((R_si @ combined_data[['a_x','a_y','a_z']].values.reshape(-1,3,1)).squeeze(), label=['ax','ay','az'])
        plt.title("Accelerometer in S-frame")
        plt.grid()
        plt.legend()
        plt.savefig(f'./out_vdr/{task["mode"]}_{frequency}/{sample_id}_acc_in_s_frame.png')
        plt.close('all')


    # TODO try use WMM to compute mount angle

    # Step 3: computing mount angle
    # keeping only heading angles to project PVT speed from ENU to S-frame with some unknown mount angle
    euler_sw = - euler_wi.copy() # here have minus sign because the rotation is opposite
    euler_sw[:,1:] = 0
    R_sw = quaternion.as_rotation_matrix(quaternion.from_euler_angles(euler_sw))

    # velocity in ENU frame
    ve = combined_data.speed.values*np.cos(combined_data.bearing.values)
    vn = combined_data.speed.values*np.sin(combined_data.bearing.values)
    vu = np.zeros_like(ve)

    if visualize:
        start_idx = 200
        end_idx = 10000 + start_idx
        step = frequency*4
        arrow_length = 100

        plt.plot(e[start_idx:end_idx], n[start_idx:end_idx],label='gt trajectory')
        for i in range(start_idx, end_idx, step):
            plt.arrow(e[i],n[i],
            arrow_length*np.cos(combined_data.bearing[i]),
            arrow_length*np.sin(combined_data.bearing[i]),
            color='red')
        plt.grid()
        plt.title(f"{task['sample_id']}")
        plt.axis('equal')
        plt.savefig(f'./out_vdr/{task["mode"]}_{frequency}/{sample_id}_gt_trjectory_with_speed_direction.png')
        plt.close('all')

        plt.plot(ve, label='gt ve')
        plt.plot(vn, label='gt vn')
        plt.title(f"{task['sample_id']}")
        plt.grid()
        plt.legend()
        plt.savefig(f'./out_vdr/{task["mode"]}_{frequency}/{sample_id}_gt_velocity_enu.png')
        plt.close('all')

    v_enu = np.vstack((ve, vn, vu)).transpose().reshape(-1,3,1)

    v_s = (R_sw @ v_enu).squeeze()
    # now velocity projected to some horizontal moving frame
    # need estimate how it is aligned relatie to vehicle body

    vs_x_min = np.inf
    best_arg = -1
    r_z = None
    yaw_mount = None

    errors = [None]*360
    forward_speeds = [None]*360
    print("Probing mount yaw angle...")
    for i in range(360):
        # brute forcing all mount angles - yaw angle between vehicle frame and S-frame
        # vehicle frame - X - to the right, Y - forward, Z- up
        yaw_mount = i*2*np.pi/360.0
        R_yaw = quaternion.as_rotation_matrix(quaternion.from_euler_angles([yaw_mount,0,0]))

        # looking for minimal median lateral velocity (projectino on X-axis) in vehicle frame  + positive median forward motion (Y-axis)
        v_v = R_yaw @ v_s.reshape(-1,3,1) # projection on X-axis
        errors[i] = np.median(np.abs(v_v)[:,0],axis=0)
        forward_speeds[i] = np.median(v_v, axis=0)[1]

        if (errors[i] < vs_x_min) and  (forward_speeds[i]> 0):
            vs_x_min = errors[i]
            best_arg = i
            best_R_yaw = R_yaw
            best_yaw_mount = yaw_mount

    v_v = (best_R_yaw @ v_s.reshape(-1,3,1)).squeeze()

    if visualize:
        plt.plot(v_v,label=['v_vehicle_x', 'v_vehicle_y','v_vehicle_z'])
        plt.title(f"PVT velocity in vehicle frame\nMount angle: {best_yaw_mount}")
        plt.grid()
        plt.legend()
        plt.savefig(f'./out_vdr/{task["mode"]}_{frequency}/{sample_id}_pvt_in_vehicle_frame.png')
        plt.close('all')


        plt.figure()
        plt.title(f'Median lat and forward velocities in vehicle frame\nMount angle: {best_yaw_mount:.4f}')
        plt.plot(np.linspace(0,2*np.pi, 360), errors, label='median lat velocity')
        plt.plot(np.linspace(0,2*np.pi, 360), forward_speeds, label='median forward velocity')
        plt.vlines(best_yaw_mount,-10, 10, color='red', label='best yaw mount angle')
        plt.grid()
        plt.legend()
        plt.savefig(f'./out_vdr/{task["mode"]}_{frequency}/{sample_id}_median_velocities_vs_mount_angle.png')
        plt.close('all')


    v_s_adjusted = R_sw @ v_enu
    v_s_adjusted = (best_R_yaw.T @ v_s_adjusted).squeeze()

    if visualize:
        plt.plot(v_s_adjusted,label=['v_s_x', 'v_s_y','v_s_z'])
        plt.title(f"PVT velocity in S-frame\nMount angle: {best_yaw_mount}")
        plt.grid()
        plt.legend()
        plt.savefig(f'./out_vdr/{task["mode"]}_{frequency}/{sample_id}_pvt_in_s_frame.png')
        plt.close('all')


    

    # Step 4:
    # rotating data to S-frame - IMU and ground truth speed 
    # putting everything together into single table
    rotate_data = combined_data.copy()


    # rotation matrix form local frame to S-frame for IMU projection
    R_si = quaternion.as_rotation_matrix(quaternion.from_euler_angles(euler_si))

    acc_s = (R_si @ combined_data[['a_x','a_y','a_z']].values.reshape(-1,3,1)).squeeze()
    gyro_s = (R_si @ combined_data[['w_x','w_y','w_z']].values.reshape(-1,3,1)).squeeze()


    rotate_data[['a_s_x', 'a_s_y','a_s_z']] = 0
    rotate_data[['a_s_x', 'a_s_y','a_s_z']] = acc_s

    rotate_data[['w_s_x', 'w_s_y','w_s_z']] = 0
    rotate_data[['w_s_x', 'w_s_y','w_s_z']] = gyro_s

    rotate_data['v_vehicle_x'] = v_v[:,0]
    rotate_data['v_vehicle_y'] = v_v[:,1]
    rotate_data['mount_yaw'] = best_yaw_mount

    rotate_data[['q_si_w','q_si_x','q_si_y','q_si_z']] = 0
    rotate_data[['q_si_w','q_si_x','q_si_y','q_si_z']] = quaternion.as_float_array(quats_si)

    rotate_data[['q_wi_w','q_wi_x','q_wi_y','q_wi_z']] = 0
    rotate_data[['q_wi_w','q_wi_x','q_wi_y','q_wi_z']] = quaternion.as_float_array(quats_wi)

    rotate_data['gt_vel_x'] = v_s_adjusted[:,0]
    rotate_data['gt_vel_y'] = v_s_adjusted[:,1]

    rotate_data.to_pickle(rotated_combined_data_path)


if __name__ == "__main__":
    start_time = time.time()

    
    dataloader_params = {
        "window" : 8*125*10,
        "step" : 1000,
        "frequency": 100,
        "batch_size" : 3
    }

    data_path = "./data/smartphone-decimeter-2022/"

    tasks = get_tasks_for_dataset(data_path)

    print('Preprocessing tasks...')
    with Pool(6) as p:
        # generating combined_data
        res = [p.apply_async(generate_combined_data,args=(t,dataloader_params['frequency'])) for t in tasks]
        for r in tqdm(res):
            r.get()

    with Pool(6) as p:
        # rotating data
        res = [p.apply_async(rotate_data,args=(t,dataloader_params['frequency'])) for t in tasks]
        for idx,r in tqdm(enumerate(res),total=len(tasks)):
            r.get()

    # dataset = GSDC_dataset('train',tasks, data_path, **dataloader_params)
    dataset = GSDC_dataset('test',tasks, data_path, **dataloader_params)

    print("--- %s seconds ---" % (time.time()-start_time))

    print("Dataset length:", dataset.__len__())
    for d in tqdm(dataset):
        # print(d)
        pass

    dataloader = DataLoader(dataset, batch_size=2, drop_last= True, collate_fn=zero_padding_collate)

    for d in tqdm(dataloader):
        print(d)