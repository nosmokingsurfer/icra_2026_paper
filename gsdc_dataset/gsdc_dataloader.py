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

import torch
from torch.utils.data import Dataset
from multiprocessing import Pool
import pymap3d as pm
import ahrs

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
            rotated_combined_data_path = './out_vdr/'+ f"{task['mode']}/" + task['sample_id'] + "_rotated_combined_data.pickle"
            tmp = pd.read_pickle(rotated_combined_data_path)
            task['length'] = len(tmp)
            number_of_steps = np.int64(np.floor(len(tmp)- self.dataloader_params['window'])/self.dataloader_params['step'])
            self.global_index_accumulator += number_of_steps
            self.task_indexes[idx] = [self.global_index_accumulator, task['sample_id'],number_of_steps]
            self.task_data[task['sample_id']] = tmp


    def __len__(self):
        return self.global_index_accumulator

    def __getitem__(self, idx):
        window = self.dataloader_params['window']
        step = self.dataloader_params['step']

        task_idx = np.searchsorted(np.array(self.task_indexes)[:,0].astype(np.int64),idx,side='right')
        global_idx, sample_id, num_steps = self.task_indexes[task_idx]

        local_slice_idx = np.int64(idx - (global_idx - num_steps))
        
        start_idx = np.int64(local_slice_idx*step)
        end_idx = np.int64(start_idx + window)

        data = self.task_data[sample_id].iloc[start_idx:end_idx]

        quats_sw = data[['q_iw_w','q_iw_x','q_iw_y','q_iw_z']].values
        euler_sw = quaternion.as_euler_angles(quaternion.from_float_array(quats_sw))


        result = {
            "acc" : torch.tensor(data[['a_s_x','a_s_y','a_s_z']].values, dtype=torch.float32),
            "gyro" : torch.tensor(data[['w_s_x','w_s_y','w_s_z']].values, dtype=torch.float32),
            "gt_velocity" : torch.tensor(data[['gt_vel_x','gt_vel_y']].values, dtype=torch.float32),
            "gt_traj" : torch.tensor(data[['e','n']].values, dtype=torch.float32),
            "yaw_angle" : torch.tensor(euler_sw[:,0], dtype=torch.float32)
        }

        # if len(result['acc'].shape) == 2:
        #     for k,v in result.items():
        #         result[k] = v.unsqueeze(0)


        if self.mode == "train":
            pass
            # apply augmentatnions here

        return result

def generate_combined_data(task, frequency=50):

    combined_data_path = './out_vdr/' + f"{task['mode']}/"+ task['sample_id'] + "_combined_data.pickle"
    if os.path.exists(combined_data_path):
        return

    if not os.path.exists('./out_vdr/'):
        os.makedirs('./out_vdr',exist_ok=True)
    
    if not os.path.exists(f"./out_vdr/{task['mode']}"):
        os.makedirs(f"./out_vdr/{task['mode']}",exist_ok=True)

    df = pd.read_csv(task['imu_file'])
    
    acc = df[df['MessageType'] == "UncalAccel"]
    acc = acc[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ']]
    acc.columns = ['t','a_x','a_y','a_z']
    acc['t'] = acc['t'].values/1e+3

    gyro = df[df['MessageType'] == "UncalGyro"]
    gyro = gyro[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ','BiasX','BiasY','BiasZ']]
    gyro.columns = ['t','w_x','w_y','w_z','bias_x','bias_y','bias_z']
    gyro['t'] = gyro['t'].values/1e+3

    gt_df = pd.read_csv(task['gt_file'])
    gt_df = gt_df[['UnixTimeMillis','LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters', 'SpeedMps', 'BearingDegrees']]
    gt_df.columns = ['t','lat','lon','alt','speed','bearing']
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


def rotate_data(task):
    sample_id = task['sample_id']

    rotated_combined_data_path = './out_vdr/'+ f"{task['mode']}/" + task['sample_id'] + "_rotated_combined_data.pickle"
    if os.path.exists(rotated_combined_data_path):
        return
    
    combined_data_path = './out_vdr/'+ f"{task['mode']}/" + task['sample_id'] + "_combined_data.pickle"

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

    # step 2:
    # computing attitude of IMU in local fixed frame:
    quats_iw = ahrs.filters.Madgwick(
        gyr= combined_data[['w_x','w_y','w_z']].values,
        acc = combined_data[['a_x','a_y','a_z']].values,
        frequency = 50.).Q

    quats_iw = quaternion.from_float_array(quats_iw)

    # Array with the 4 elements of quaternion of the form [w, x, y, z]
    plt.plot(np.unwrap(quaternion.as_euler_angles(quats_iw),axis=0),label=['yaw','pitch','roll'])
    plt.title('Euler angles from local fixed frame to IMU frame')
    plt.grid()

    plt.plot(np.unwrap(combined_data.bearing/180*np.pi - np.pi) + np.pi,'-',color='red', label='bearing from pvt')
    plt.legend()
    plt.savefig(f'./out_vdr/{task["mode"]}/{sample_id}_euler_w_to_imu.png')
    plt.close('all')
    # plt.show()

    # yaw pitch roll from some fixed frame to IMU
    imu_euler = np.unwrap(quaternion.as_euler_angles(quats_iw),axis=0)

    # keeping only pitch and roll angles to rotate IMU to S-frame
    s_euler = -imu_euler.copy() # here we have minus sign - need rotation FROM imu frame to S-frame
    s_euler[:,0] = 0
    quats_sw = quaternion.from_euler_angles(s_euler)

    # Step 3: computing mount angle
    # keeping only heading angles to project PVT speed from ENU to S-frame with some unknown mount angle
    yaw_euler = imu_euler.copy()
    yaw_euler[:,1:] = 0

    ve = combined_data.speed.values*np.cos(combined_data.bearing.values*np.pi/180.)
    vn = combined_data.speed.values*np.sin(combined_data.bearing.values*np.pi/180.)
    vu = np.zeros_like(ve)

    R_sw_yaw = quaternion.as_rotation_matrix(quaternion.from_euler_angles(yaw_euler))

    v = np.vstack((ve,vn,vu)).transpose().reshape(-1,3,1)
    # now velocity projected to some horizontal moving frame
    # need estimate how it is aligned relatie to vehicle body

    vs = (R_sw_yaw @ v).squeeze()
    vs_x = vs[:,0]

    vs_y_min = np.inf
    best_arg = -1
    r_z = None
    yaw_mount = None

    errors = [None]*360
    forward_speeds = [None]*360
    print("Probing mount yaw angle...")
    for i in range(360):
        yaw_mount = i*2*np.pi/360.0

        r_z = quaternion.as_rotation_matrix(quaternion.from_euler_angles([yaw_mount,0,0]))

        # plt.plot((r_z @ vs.reshape(-1,3,1)).squeeze()[:,1])
        # vehicle frame - X - to the right, Y - forward, Z- up
        # looking for minimal median lateral velocity (projectino on X-axis) in vehicle frame  + positive median forward motion (Y-axis)
        cur = np.median(np.abs(r_z @ vs.reshape(-1,3,1)).squeeze()[:,0]) # projection on X-axis
        errors[i] = cur
        forward_speeds[i] = np.median(r_z @ vs.reshape(-1,3,1), axis=0)[1]

        if (cur < vs_y_min) and  (forward_speeds[i]> 0):
            vs_y_min = cur
            best_arg = i
            best_r_z = r_z
            best_yaw_mount = yaw_mount

    v_vehicle = (best_r_z@vs.reshape(-1,3,1)).squeeze()

    ve_s = combined_data.speed.values*np.cos(combined_data.bearing.values*np.pi/180.)
    vn_s = combined_data.speed.values*np.sin(combined_data.bearing.values*np.pi/180.)
    v_s = np.vstack((ve_s,vn_s)).transpose().reshape(-1,2,1)

    plt.plot(v_vehicle,label=['v_vehicle_x', 'v_vehicle_y','v_vehicle_z'])
    plt.title(f"PVT velocity in vehicle frame\nMount angle: {best_yaw_mount}")
    plt.grid()
    plt.savefig(f'./out_vdr/{task["mode"]}/{sample_id}_pvt_in_vehicle_frame.png')
    plt.close('all')

    plt.figure()
    plt.title('Median lat and forward velocities in vehicle frame')
    plt.plot(np.linspace(0,2*np.pi, 360), errors, label='median lat velocity')
    plt.plot(np.linspace(0,2*np.pi, 360), forward_speeds, label='median forward velocity')
    plt.grid()
    plt.legend()
    plt.savefig(f'./out_vdr/{task["mode"]}/{sample_id}_median_velocities_vs_mount_angle.png')
    plt.close('all')

    # Step 4:
    # rotating data to S-frame - IMU and ground truth speed 
    # putting everything together into single table
    rotate_data = combined_data.copy()


    # rotation matrix form local frame to S-frame for IMU projection
    R_sw = quaternion.as_rotation_matrix(quaternion.from_euler_angles(s_euler))

    acc_s = (R_sw @ combined_data[['a_x','a_y','a_z']].values.reshape(-1,3,1)).squeeze()
    gyro_s = (R_sw @ combined_data[['w_x','w_y','w_z']].values.reshape(-1,3,1)).squeeze()


    rotate_data[['a_s_x', 'a_s_y','a_s_z']] = 0
    rotate_data[['a_s_x', 'a_s_y','a_s_z']] = acc_s

    rotate_data[['w_s_x', 'w_s_y','w_s_z']] = 0
    rotate_data[['w_s_x', 'w_s_y','w_s_z']] = gyro_s

    rotate_data['v_vehicle_x'] = v_vehicle[:,0]
    rotate_data['v_vehicle_y'] = v_vehicle[:,1]
    rotate_data['mount_yaw'] = best_yaw_mount

    rotate_data[['q_sw_w','q_sw_x','q_sw_y','q_sw_z']] = 0
    rotate_data[['q_sw_w','q_sw_x','q_sw_y','q_sw_z']] = quaternion.as_float_array(quats_sw)

    rotate_data[['q_iw_w','q_iw_x','q_iw_y','q_iw_z']] = 0
    rotate_data[['q_iw_w','q_iw_x','q_iw_y','q_iw_z']] = quaternion.as_float_array(quats_iw)

    rotate_data['gt_vel_x'] = np.sin(-best_yaw_mount)*combined_data.speed.values
    rotate_data['gt_vel_y'] = np.cos(-best_yaw_mount)*combined_data.speed.values

    plt.plot(acc_s)
    plt.title("Accelerometer in S-frame")
    plt.grid()
    plt.savefig(f'./out_vdr/{task["mode"]}/{sample_id}_acc_in_s_frame.png')
    plt.close('all')

    rotate_data.to_pickle(rotated_combined_data_path)


if __name__ == "__main__":
    start_time = time.time()

    
    dataloader_params = {
        "window" : 8*125*10,
        "step" : 1000,
        "frequency": 50,
        "batch_size" : 3
    }

    data_path = "./data/smartphone-decimeter-2022/"
    imu_files = list(Path(data_path + "train/").rglob("**/device_imu.csv"))


    print("Indexing tasks...")
    tasks = []
    for t in tqdm(imu_files):
        sample_id = t.parts[-3].replace('-','_') + "_" + t.parts[-2]
        imu_file = str(t.parent / "device_imu.csv")
        gt_file = str(t.parent / "ground_truth.csv")

        tasks.append(
            {
                "sample_id" : sample_id,
                "mode" : "train",
                "imu_file" : imu_file,
                "gt_file" : gt_file
            }
        )


    print('Preprocessing tasks...')
    with Pool(6) as p:
        # generating combined_data
        res = [p.apply_async(generate_combined_data,args=(t,)) for t in tasks]
        for r in tqdm(res):
            r.get()

    with Pool(6) as p:
        # rotating data
        res = [p.apply_async(rotate_data,args=(t,)) for t in tasks]
        for idx,r in tqdm(enumerate(res),total=len(tasks)):
            r.get()

    train_dataset = GSDC_dataset('train',tasks, data_path, **dataloader_params)
    # test_dataset = GSDC_dataset('test', **dataloader_params)

    print("--- %s seconds ---" % (time.time()-start_time))

    print("Dataset length:", train_dataset.__len__())
    for d in tqdm(train_dataset):
        # print(d)
        pass