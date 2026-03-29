from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class GSDC_dataset(Dataset):
    def __init__(self, data_path = "./data/smartphone-decimeter-2022/", **args):
        self.dataloader_params = args
        assert "window" in self.dataloader_params

        self.data_path = data_path

        train_files = list(Path(data_path + "train/").rglob("**/device_imu.csv"))
        test_files = list(Path(data_path + "test/").rglob("**/device_imu.csv"))

        print("Indexing training tasks...")
        self.train_tasks = []
        for t in tqdm(train_files):
            sample_id = t.parts[-3].replace('-','_') + "_" + t.parts[-2]
            imu_file = str(t.parent / "device_imu.csv")
            gt_file = str(t.parent / "ground_truth.csv")

            self.train_tasks.append(
                {
                    "sample_id" : sample_id,
                    "mode" : "train",
                    "imu_file" : imu_file,
                    "gt_file" : gt_file
                }
            )
        
        print(len(self.train_tasks))

        for t in self.train_tasks:
            self.map_indexes(t)


        print("Indexing test tasks...")
        self.test_tasks = []

        for t in tqdm(test_files):
            sample_id = t.parts[-3].replace('-','_') + "_" + t.parts[-2]
            imu_file = str(t.parent / "device_imu.csv")
            

            self.test_tasks.append(
                {
                    "sample_id" : sample_id,
                    "mode" : "test",
                    "imu_file" : imu_file,
                    "gt_file" : gt_file
                }
            )


        print(len(self.test_tasks))

    def map_indexes(self, task):

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

        frequency = self.dataloader_params['frequency']
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

        gt_resampled = pd.DataFrame(gt_resampled,columns = gt_df.columns)

        combined_data = pd.concat((acc_resampled, gyro_resampled, gt_resampled),axis=1)

        print(combined_data.shape)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    dataloader_params = {
        "window" : 2000,
        "step" : 50,
        "frequency": 50
    }

    dataset = GSDC_dataset(**dataloader_params)