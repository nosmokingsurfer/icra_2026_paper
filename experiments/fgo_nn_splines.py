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

def compute_delta_x(gt_poses, estimated_poses):
    gt_poses_se3 = [mrob.SE3(gt_poses[i]) for i in range(len(gt_poses))]
    estimated_poses_se3 = [mrob.SE3(estimated_poses[j]) for j in range(len(estimated_poses))]
    
    result_Ln = [(gt_poses_se3[k] * estimated_poses_se3[k].inv()).Ln() for k in range(len(gt_poses_se3))]
    return np.array(result_Ln)


def integrate_pred_vel(pred_vel, gt_poses, gt_pose0, dt=0.01):
    T = pred_vel.shape[0]
    poses = torch.zeros((T, 3), dtype=pred_vel.dtype, device=pred_vel.device)

    x, y, yaw = gt_pose0[0], gt_pose0[1], gt_pose0[2]
    poses[0] = torch.stack([x, y, yaw])

    for t in range(1, T):
        vx, vy = pred_vel[t - 1]

        x = x + vx * dt
        y = y + vy * dt
        yaw = gt_poses[t, 2]  # directly use GT yaw

        poses[t] = torch.stack([x, y, yaw])

    return poses


def populate_graph(pred_poses, gt_poses):
    graph = mrob.FGraphDiff()

    W_odo = torch.eye(6) * 10.0
    W_gps = torch.eye(6) * 1.0

    node_ids = []
    T_nodes = []
    T = pred_poses.shape[0]
    
    # Initialize nodes using predicted poses (world frame)
    for i in range(T): # неподвижная система координат
        pose_i = pred_poses[i].detach().cpu().numpy()
        T_i = mrob.SE3(convert_to_se3(pose_i))
        T_nodes.append(T_i)
        node_id = graph.add_node_pose_3d(T_i)
        node_ids.append(node_id)

    # Odometry factors from predicted poses
    for i in range(len(node_ids) - 1):
        T_pred_i   = T_nodes[i]
        T_pred_ip1 = T_nodes[i + 1]
        T_odo = T_pred_i.inv() * T_pred_ip1 #mrob.SE3(pred_v, w (from IMU))
        graph.add_factor_2poses_3d_diff(T_odo, node_ids[i], node_ids[i + 1], W_odo.numpy())

    # GPS factors from GT poses
    for i in range(len(node_ids)):
        # if i % 10 == 0 or i == len(node_ids) - 1:
        T_gps = mrob.SE3(convert_to_se3(gt_poses[i].detach().cpu().numpy()))
        graph.add_factor_1pose_3d_diff(T_gps, node_ids[i], W_gps.numpy())
    
    return graph


def print_2d_graph(graph, gt_poses):
    x = np.array(graph.get_estimated_state())
    
    fig = plt.figure()

    plt.plot(x[:,0,3],x[:,1,3], '-b', marker='o', label='estimated')
    plt.plot(gt_poses[:, :2][:, 0], gt_poses[:, :2][:, 1], label='GT', color='red')
    plt.title("2D Pose Graph")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')
    plt.grid()
    # plt.show()
    return fig


def make_gif_from_figures(folder_path, output_path="graph_evolution.gif", num_epochs=100, duration=0.3):
    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.startswith("poses_") and f.endswith(".png") ],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    
    file_list = file_list[0:num_epochs]

    images = [imageio.imread(os.path.join(folder_path, fname)) for fname in file_list]
    imageio.mimsave(output_path, images, duration=duration)
    print(f"GIF saved to {output_path}")


def plot_integrated_vel(poses_1, poses_2):
    plt.figure()
    plt.plot(poses_1[:, 0].detach(), poses_1[:, 1].detach(), marker='o', label='integrated')
    plt.plot(poses_2[:, 0].detach(), poses_2[:, 1].detach(), marker='o', label='gt')
    plt.legend()
    plt.show()
    return

def process_one_graph(vel_pred, gt_pose_seq, dt):
    S = vel_pred.shape[0]
    # Step 1: integrate vel_pred[b] into poses
    pred_poses = integrate_pred_vel(vel_pred, gt_pose_seq, gt_pose_seq[0], dt=dt)  # [S, 3]
    
    # Step 2: build & solve graph
    graph = populate_graph(pred_poses, gt_pose_seq)  # FGraphDiff
    graph.build_jacobians()
    graph.solve(mrob.FGraphDiff_LM, maxIters=100)
    dL_dz = graph.get_dx_dz() / dt  # [6S, 6S + (S-1)*S]

    # Step 3: compute delta_x
    gt_poses_se3 = convert_to_se3(gt_pose_seq.cpu().numpy())
    delta_x = compute_delta_x(gt_poses_se3, graph.get_estimated_state())  # [S, 6]
    delta_x_flat = delta_x.reshape(-1)  # [6S]

    # Step 4: compute gradient per sample and stack
    grad = (delta_x_flat @ dL_dz)[:S*6].reshape(-1, 6)
    grad_final = torch.from_numpy(grad[:, [3, 4]]).float()  # [S, 2]

    # Step 5: computing chi2 and rmse errors
    chi2 = graph.chi2()
    rmse = compute_rmse_and_yaw(graph, gt_poses_se3, delta_x, plot=False)
    
    return grad_final, chi2, rmse

def run_validation(epoch, output_path, model, dataset, num_traj):
    model.eval()

    assert num_traj <= len(dataset)

    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)

    for i in range(num_traj):
        result = {}

        sample  = dataset.__getitem__(i)
        imu_seq = sample['noisy_imu']
        vel_seq = sample['gt_vel']
        gt_poses_seq = sample['gt_poses']
        dt = dataset.step/dataset.sampling_rate

        S, C, W = imu_seq.shape
        vel_pred = model(imu_seq.reshape(-1, C, W)).reshape(S, -1)
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
        plt.savefig(f'{output_path}idx_{i}_epoch_{epoch}.jpg')
        plt.close('all')
        pickle.dump(result, open(f'{output_path}idx_{i}_epoch_{epoch}_traj.pkl','wb'))


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

    output_path = f"./out/graphs_seq_{subseq_len}_epochs_{n_epochs}/"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

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
    )
    
    # model.load_state_dict(torch.load("out/model_fgo.pth"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    criterion = nn.MSELoss()

    path_to_splines =  './out/splines'

    number_of_splines = 20
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)
        
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
        run_validation(epoch, output_path, model, val_dataloader.dataset, trajectories_to_save)
            
        total_chi2, total_rmse = 0.0, 0.0
        model.train()
        
        for sample in tqdm(train_dataloader, position=0, leave=True):
            # sample  = dataset.__getitem__(i)
            imu_seq = sample['noisy_imu']
            vel_seq = sample['gt_vel']
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
                    res = [p.apply_async(process_one_graph, args=(vel_pred[b].detach(), gt_poses_seq[b].detach(), dt)) for b in range(B)]


                    for i, r in enumerate(res):
                        all_grads[i], chi2, rmse = r.get()
                        total_chi2 += chi2
                        total_rmse += rmse

                grad_tensor = torch.stack(all_grads)  # [B, S, 2]
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

    # for s in range(1,5,1):
    #     run_spline_experiment(s, 50)

    run_spline_experiment(1, 100)
    run_spline_experiment(6, 100)