# noisy_imu -> diffusion -> denoised_imu -> ronin -> velocity -> FGO -> trajectory -> loss 
# fgo nn splines input size before reshape torch.Size([64, 6, 3, 100]) and after reshape torch.Size([384, 3, 100])
#                                                       B, S, C, W                                  -1, C, W
# diffusion splines input and output size torch.Size([64, 3, 100])
import mrob
import numpy as np
np.set_printoptions(precision=4,linewidth=180)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import imageio.v2 as imageio

import os
import time
import sys
import pickle
from tqdm import tqdm
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.multiprocessing import Pool
from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset, convert_to_se3

from experiments.utils_metrics import compute_rmse_and_yaw, compute_ate_rte

from metric import compute_ate_rte
from ronin_resnet import get_model
from ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule
from model_temporal import TCNSeqNetwork
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.diffusion_splines import IMUDenoiser
from ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule
from experiments.fgo_nn_splines import run_validation,process_one_graph

class DiffussionNN(nn.Module):
    def __init__(self):
        super().__init__() 

        diffusion_config = {
            'spline_path': './out/splines',  # or path to your spline files
            'num_samples': 5000,  # if using synthetic data
            'window_size': 100,
            'noise_level': 0.2,
            'stride' : 20,
            'imu_freq': 100.0,
            'batch_size': 64,
            'num_workers': 0,
            'num_epochs': 100,
            'val_interval': 5,
            'T': 1000,  # diffusion timesteps
            'lr': 1e-4,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        trained_model_state = torch.load(open('model.pt','rb'), self.device)
        self.diffusion_model =  IMUDenoiser(
            input_channels=3,  # ax, ay, ωz
            window=diffusion_config['window_size']
        ).to(self.device)
        print(self.device)
        self.diffusion_model.load_state_dict(trained_model_state)

        self.nn_model = ResNet1D(
            num_inputs=3,          # Must match diffusion's output channels
            num_outputs=2,        # Your desired output classes/dimension
            block_type=BasicBlock1D,   # Or Bottleneck1D
            group_sizes=[2, 2, 2], # Example architecture (adjust as needed)
            base_plane=64          # Standard ResNet starting plane size
        )
        self.nn_model = torch.load(open('out/graphs_seq_1_epochs_200_baseline/model_epoch_100.cpt','rb'), weights_only=False).to(self.device)
        # self.nn_model.load_state_dict(trained_model_state)


    def forward(self, x):
        # model(denoising_model(imu.reshape(-1,3,w))).reshape(B,S,-1)
        # TODO: think about amount of diffusion steps
        inference_t = torch.full((x.shape[0],), 15).to(self.device)
        x = self.diffusion_model(x, t=inference_t)
        x = self.nn_model(x)
        return x
    

def run_spline_experiment(subseq_len = 3, n_epochs=300):
    '''
    Odometry model training pipeline on spline dataset
    if subseq_len == 1 - conventional window-based training mode
    if subseq_len > 1 - FGO loss training mode
    '''

    results = {}
    results['n_epochs'] = n_epochs
    results['n_actual_epochs'] = 0
    results['subseq_len'] = subseq_len

    output_path = f"./out/graphs_seq_{subseq_len}_epochs_{n_epochs}_diffusion_for_comparison/"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffussionNN().to(device)
    
    # model.load_state_dict(torch.load("out/model_fgo.pth"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    criterion = nn.MSELoss()

    path_to_splines =  './out/splines_fixed'

    # number_of_splines = 20
    # if not os.path.exists(path_to_splines):
    #     number_of_control_nodes = 10
    #     generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)
        
    window_size=100
    step_size=10
    sampling_rate =100
    
    dataset = Spline_2D_Dataset(path_to_splines, 
                                window=window_size,
                                sampling_rate=100,
                                subseq_len=subseq_len,
                                mode='regression',
                                enable_noise= not True)

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=dataset.get_collate_fn())

    val_dataset = Spline_2D_Dataset(path_to_splines,
                                window=window_size,
                                subseq_len=89,
                                mode='regression',
                                enable_noise= not True)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    dt = step_size/sampling_rate
    rmse_errors = []
    chi2_errors = []
    learning_rates = []
    
    trajectories_to_save = 3
    results['num_val_traj'] = trajectories_to_save

    for epoch in range(n_epochs):

        # running validation every epoch
        run_validation(epoch, output_path, model, val_dataloader.dataset, trajectories_to_save, model.device)
            
        total_chi2, total_rmse = 0.0, 0.0
        model.train()
        
        for sample in tqdm(train_dataloader, position=0, leave=True):
            # sample  = dataset.__getitem__(i)
            imu_seq = sample['noisy_imu'].to(model.device)
            vel_seq = sample['gt_vel'].to(model.device)
            gt_poses_seq = sample['gt_poses']
            
            # imu_seq: [B, S, 3, W]
            # vel_seq: [B, S, 2]
            B, S, C, W = imu_seq.shape

            assert S == subseq_len

            # running model inference for all slices at once
            vel_pred = model(imu_seq.reshape(-1, C, W)).reshape(B, S, -1)
            
            optimizer.zero_grad()


            if subseq_len > 1:
                all_grads = [None for _ in range(B)]

                total_chi2 = 0
                total_rmse = 0

                with Pool(8) as p:
                    res = [p.apply_async(process_one_graph, args=(vel_pred[b].detach().cpu(), gt_poses_seq[b].detach().cpu(), dt)) for b in range(B)]


                    for i, r in enumerate(res):
                        all_grads[i], chi2, rmse = r.get()
                        total_chi2 += chi2
                        total_rmse += rmse

                grad_tensor = torch.stack(all_grads).to(model.device)  # [B, S, 2]
                vel_pred.backward(gradient= - grad_tensor)

            else:
                loss = torch.linalg.norm(vel_pred - vel_seq)
                loss.backward()

                total_rmse += loss.detach().cpu().item()
                total_chi2 = np.nan

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
        scheduler.step()
        
        chi2_errors.append(total_chi2 / len(train_dataloader))
        rmse_errors.append(total_rmse / len(train_dataloader))
        learning_rates.append(scheduler.get_last_lr()[0])

        print(f"[Epoch {epoch}] Chi2: {chi2_errors[-1]:.4f}, RMSE: {rmse_errors[-1]:.4f}")
        print(f"Learning Rate : {scheduler.get_last_lr()}")
        results['n_actual_epochs'] += 1

        results['chi2_errors'] = chi2_errors
        results['rmse_errors'] = rmse_errors
        results['learning_rate'] = learning_rates
        pickle.dump(results, open(output_path+'results.pkl','wb'))

        if epoch % 10 == 0:
            torch.save(model, output_path + f'model_epoch_{epoch}.cpt')

    plt.title('Errors: CHi2 and RMSE')
    plt.plot(chi2_errors,label='chi2')
    plt.plot(rmse_errors,label='rmse')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    plt.savefig(f'{output_path}/errors.png')
    plt.close('all')

    plt.figure()
    plt.plot(learning_rates,label='learning rate')
    plt.grid()
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'{output_path}/learning_rate.png')
    plt.close('all')


if __name__ == "__main__":

    st = time.time()
    run_spline_experiment(1, 200)
    end_t = time.time() - st
    print("No graph loss time:   ", end_t, "\n\n\n")
    st = time.time()
    run_spline_experiment(6, 200)
    end_t = time.time() - st
    print("With graph loss time:   ", end_t, "\n\n\n")