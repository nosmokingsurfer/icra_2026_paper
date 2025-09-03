import mrob
import numpy as np
np.set_printoptions(precision=4,linewidth=180)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import imageio.v2 as imageio

import os
import sys
import pickle
from tqdm import tqdm
import shutil


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.multiprocessing import Pool
from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset, convert_to_se3

from experiments.utils_metrics import compute_rmse_and_yaw, compute_ate_rte, save_file_split

from metric import compute_ate_rte
from ronin_resnet import get_model
from ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule
from model_temporal import TCNSeqNetwork

from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
from spline_dataset.spline_dataloader import Spline_2D_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.multiprocessing import Pool
import numpy as np
np.set_printoptions(precision=4,linewidth=180)
import pickle
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from ronin_resnet import get_model
from ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule
from fgo.fgo_implementation import process_one_graph, integrate_pred_vel, populate_graph
from metric import compute_ate_rte, save_file_split

def run_validation(epoch, output_path, model, dataset, num_traj, device):
    model.eval()

    assert num_traj <= len(dataset)


    for i in range(num_traj):
        result = {}

        sample  = dataset.__getitem__(i)
        imu_seq = sample['noisy_imu'].to(device)
        gt_poses_seq = sample['gt_poses']
        dt = dataset.step/dataset.sampling_rate

        trajectories_path = output_path / "trajectories"
        trajectories_path.mkdir(parents=True, exist_ok=True)

        S, C, W = imu_seq.shape
        vel_pred = model(imu_seq.reshape(-1, C, W)).reshape(S, -1).detach().cpu()
        pred_poses = integrate_pred_vel(vel_pred, gt_poses_seq, gt_poses_seq[0], dt=dt)  # [S, 3]
        graph = populate_graph(pred_poses, gt_poses_seq)
        # graph.solve(mrob.FGraphDiff_LM, maxIters=100)
        # print_2d_graph(graph,gt_poses_seq)
        est_pose_seq = np.array(graph.get_estimated_state())


        result.update({'open_loop_traj' : est_pose_seq, 'gt_traj' : gt_poses_seq.detach().cpu().numpy()})
        

        est_traj = est_pose_seq[:,:2,3]
        gt_traj = gt_poses_seq[:,:2].detach().cpu().numpy()
        ate, rte = compute_ate_rte(est_traj,gt_traj,60/dt)

        result['ate'] = ate
        result['rte'] = rte

        plt.figure(figsize=(8,8))

        plt.plot(est_pose_seq[:,0,3],est_pose_seq[:,1,3], '-b', marker='o', label='estimated')
        plt.plot(gt_poses_seq[:, :2][:, 0], gt_poses_seq[:, :2][:, 1], label='GT', color='red')
        plt.title("2D Pose Graph")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.axis('equal')
        plt.grid()
        plt.title(f'Epoch: {epoch}\n' + \
            f"ATE: {ate:.3f}, RTE: {rte:.3f}")
        plt.tight_layout()
        plt.savefig(trajectories_path / f'idx_{i}_epoch_{epoch}.jpg')
        plt.close('all')
        pickle.dump(result, open(output_path / f'idx_{i}_epoch_{epoch}_traj.pkl','wb'))

def fgo_nn_splines_train_loop(train_dataloader, val_dataloader=None, subseq_len = 3, 
                          n_epochs=300, output_path=None, device="cpu", start_lr=1e-3,
                          tensorboard_dir=None):
    '''
    Odometry model training pipeline on spline dataset
    if subseq_len == 1 - conventional window-based training mode
    if subseq_len > 1 - FGO loss training mode
    '''

    results = {}
    results['n_epochs'] = n_epochs
    results['n_actual_epochs'] = 0
    results['subseq_len'] = subseq_len

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)

    model = ResNet1D(
        num_inputs=3,       
        num_outputs=2,         
        block_type=BasicBlock1D,
        group_sizes=[2, 2, 2],   
        base_plane=64,
        output_block=FCOutputModule,  
        kernel_size=3,
        fc_dim=512,             
        in_dim=7,            
        dropout=0.5,
        trans_planes=128
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    criterion = nn.MSELoss()

    step_size=10
    dt = step_size/train_dataloader.sampling_rate
    rmse_errors = []
    chi2_errors = []
    learning_rates = []
    
    trajectories_to_save = 3
    results['num_val_traj'] = trajectories_to_save

    if not tensorboard_dir:
        tensorboard_dir = "./default_tensorboard_dir"

    writer = SummaryWriter(tensorboard_dir)
    
    for epoch in range(n_epochs):

        # running validation every epoch
        if val_dataloader:
            run_validation(epoch, output_path, model, val_dataloader.dataset, trajectories_to_save, device)
        total_chi2, total_rmse = 0.0, 0.0
        model.train()
        for sample in tqdm(train_dataloader, position=0, leave=True):
            imu_seq = sample['noisy_imu'].to(device)
            vel_seq = sample['gt_vel'].to(device)
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

                grad_tensor = torch.stack(all_grads).to(device)  # [B, S, 2]
                vel_pred.backward(gradient= - grad_tensor)
            else:
                loss = criterion(vel_pred, vel_seq)
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

        writer.add_scalar("chi2_errors", chi2_errors[-1], results['n_actual_epochs'])
        writer.add_scalar("rmse_errors", rmse_errors[-1], results['n_actual_epochs'])
        writer.add_scalar("learning_rates", learning_rates[-1], results['n_actual_epochs'])

        pickle.dump(results, open(output_path+'results.pkl','wb'))

        if epoch % 10 == 0:
            torch.save(model, output_path + f'model_epoch_{epoch}.cpt')






if __name__ == "__main__":
    subseq_len=2
    n_epochs=5
    output_path = f"./out/graphs_seq_{subseq_len}_epochs_{n_epochs}_testing_subseq_length/"
    path_to_splines = "./out/splines"
    window_size=100



    train_dataset = Spline_2D_Dataset(path_to_splines, 
                                window=window_size,
                                sampling_rate=100,
                                subseq_len=subseq_len,
                                mode='regression',
                                enable_noise= not True)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.get_collate_fn())

    # There was subseq_len equal to 90. Why?
    val_dataset = Spline_2D_Dataset(path_to_splines, window=window_size, subseq_len=subseq_len, enable_noise= not True, is_val = True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_dataset.get_collate_fn())

    save_file_split(path_to_splines, output_path)

    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    fgo_nn_splines_train_loop(train_dataloader, val_dataloader=val_dataloader, subseq_len = subseq_len, 
                          n_epochs=n_epochs, output_path=output_path, device=device, start_lr=1e-3,
                          tensorboard_dir=None)