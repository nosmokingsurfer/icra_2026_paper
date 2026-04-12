import matplotlib.pyplot as plt

import theseus as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy

from gsdc_dataset.simple_vdr_model import SimpleVDRmodel
from gsdc_dataset.gsdc_dataloader import GSDC_dataset


class FGO_SimpleVDRModel(nn.Module):
    def __init__(self, **dataloader_params):
        super(FGO_SimpleVDRModel,self).__init__()
        self.model = SimpleVDRmodel()
        self.dataloader_params = dataloader_params

        # initializing theseus layer
        # optimization variables for inner loop
        window = self.dataloader_params['window']

        # have velocity output every 125 input samples
        self.B = self.dataloader_params['batch_size']
        self.N = window//125 + 1 # here 125 is deom SimpleVDR Model encoder structure

        poses : List[th.SE2] = []

        # number of poses is the same as the number of nodes in trajectory
        for i in range(self.N):
            poses.append(th.SE2(torch.zeros(self.B,3),name=f"pose_{i}"))

        # adding cost functions
        self.cost_functions = []

        # adding cost factors for odometry
        for i in range(self.N-1):
            # odometry measurmenets will depend on NN output
            # pred_acc = model(input_acc[:,i].view(B,-1))  #<====== here we attach computational graph of NN to all the odometry factors via predicting acceleration
            meas_tensor = th.SE2(torch.zeros((self.B,3)), name=f"predicted_odometry_{i}")

            # meas_tensor = th.SE2(torch.tensor([gt_traj[i+1] - gt_traj[i], 0, 0]).reshape(1,-1))
            cost_between = th.ScaleCostWeight(torch.ones((self.B,1)), name=f"scale_between_{i}")
            self.cost_functions.append(
                        th.Between(poses[i], poses[i+1], meas_tensor,
                                cost_between,
                                name=f"between_{i}"))
            
        # adding cost fuctors for absolute position
        for i in range(self.N):
            gt_pose_tensor = th.SE2(torch.zeros(self.B,3), name=f"gt_pose_{i}")
            scale_gps = th.ScaleCostWeight(torch.ones((self.B,1)), name=f"scale_gps_{i}")
            self.cost_functions.append(th.Difference(poses[i], gt_pose_tensor, scale_gps, name=f"gps_{i}"))

            self.objective = th.Objective()
        for c in self.cost_functions:
            self.objective.add(c)

        print("+"*40)
        print(f"AUX variables: {len(self.objective.aux_vars)}")
        print("+"*40)
        for c in self.objective.aux_vars:
            print(c)
        print("+"*40)

        print(f"Optimization variables: {len(self.objective.optim_vars)}")
        print("+"*40)
        for c in self.objective.optim_vars:
            print(c)
        print("+"*40)

        self.optimizer = th.LevenbergMarquardt(self.objective, th.CholmodSparseSolver)

        self.theseus_layer = th.TheseusLayer(self.optimizer)



    def forward(self, sample):
        acc= sample['acc']
        gyro = sample['gyro']
        gt_vel = sample['gt_velocity']
        gt_traj = sample['gt_traj']
        yaw_angle = sample['yaw_angle']

        window = torch.ones((2,1,8))/8
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8, groups=2).swapaxes(-1,-2)
        gt_traj = torch.conv1d(gt_traj.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)

        window=torch.ones((1,1,8))/8
        yaw_angle=torch.conv1d(yaw_angle.unsqueeze(-2),window, stride=8).swapaxes(-1,-2).squeeze()

        imu = torch.concat((acc,gyro),dim=-1).swapaxes(-1,-2)
        odometry = self.model(imu)

        B,T = yaw_angle.shape
        s,c = torch.sin(yaw_angle), torch.cos(yaw_angle)
        R = torch.zeros(B,T,2,2)
        # c -s
        # s  c
        R[:,:,0,0] = R[:,:,1,1] = c
        R[:,:,0,1] = -s
        R[:,:,1,0] = s

        odometry = (R@odometry.unsqueeze(-1)).squeeze()

        predicted_incremental_se2 = torch.concat((odometry, yaw_angle.unsqueeze(-1)),dim=-1)
        gt_traj_se2 = torch.concat((gt_traj,yaw_angle.unsqueeze(-1)),dim=-1)

        theseus_input = {
        }

        for i in range(self.N-1):
            theseus_input[f'predicted_odometry_{i}'] = th.SE2(x_y_theta=predicted_incremental_se2[:,i,:]).tensor
            

        for i in range(self.N):
            theseus_input[f'gt_pose_{i}'] = th.SE2(x_y_theta=gt_traj_se2[:,i,:].detach()).tensor
            theseus_input[f'pose_{i}'] = th.SE2(x_y_theta=deepcopy(gt_traj_se2[:,i,:].detach())).tensor

        # for k,v in theseus_input.items():
        #     print(k,v)

        self.theseus_layer.objective.update(theseus_input)

        out, status = self.theseus_layer.forward(theseus_input)

        

        return out, status





if __name__ == "__main__":

    dataloader_params = {
        "window" : 8*125*1,
        "step" : 200,
        "frequency" : 50,
        "batch_size" : 3
    }

    dataset = GSDC_dataset("train", **dataloader_params)

    model = FGO_SimpleVDRModel(**dataloader_params)

    train_dataloader = DataLoader(dataset, batch_size=dataloader_params['batch_size'], shuffle=True)

    for sample in train_dataloader:

        output  = model(sample)

        print(output)


        
