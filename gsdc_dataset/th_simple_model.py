import theseus as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        gt_velocity = sample['gt_velocity']
        gt_traj = sample['gt_traj']
        yaw_angle = sample['yaw_angle']


        imu = torch.concat((acc,gyro),dim=-1).swapaxes(-1,-2)
        odometry = self.model(imu)

        theseus_input = {
        }

        for i in range(self.N-1):
            theseus_input[f'predicted_odometry_{i}'] = odometry[:,i,:]

        for i in range(self.N):
            theseus_input[f'gt_pose_{i}'] = gt_traj[:,i]



        self.theseus_layer.forward(theseus_input)

        print(self.theseus_layer.objective)
        pass





if __name__ == "__main__":

    dataloader_params = {
        "window" : 8*125*1,
        "step" : 200,
        "frequency" : 50,
        "batch_size" : 2
    }

    dataset = GSDC_dataset("train", **dataloader_params)

    model = FGO_SimpleVDRModel(**dataloader_params)

    train_dataloader = DataLoader(dataset, batch_size=dataloader_params['batch_size'], shuffle=True)

    for sample in train_dataloader:

        output  = model(sample)

        print(output)


        
