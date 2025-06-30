import mrob
import numpy as np
np.set_printoptions(precision=4,linewidth=180)
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mrob_num_diff.graph_generator import ToRoContainer, compare_gradients
from mrob_num_diff.num_diff import find_factor_coord_idx, visualize_gradient
from mrob_num_diff.num_diff_3d import numerical_diff2_3d, numerical_diff2_3d_sparse, numerical_diff1_3d
from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset, convert_to_se3

from ronin.ronin_resnet import get_model
from ronin.ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule
from ronin.model_temporal import TCNSeqNetwork

def plot_losses(chi2_losses, rmse_losses, labels=("Chi²", "RMSE"), title="Losses", output_path=None):
    epochs = list(range(len(chi2_losses)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(epochs, chi2_losses, color='blue', marker='o', label=labels[0])
    ax1.set_ylabel(labels[0], fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.grid(True)
    ax1.legend(fontsize=12)

    ax2.plot(epochs, rmse_losses, color='red', marker='x', label=labels[1])
    ax2.set_xlabel("Epoch", fontsize=14)
    ax2.set_ylabel(labels[1], fontsize=14)
    ax2.grid(True)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close('all')


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


def plot_pose_comparison(est_xy, gt_xy, est_yaw, gt_yaw, epoch, rmse_trans, output_dir=None):
    N = len(est_xy)
    plt.figure(figsize=(10, 8))
    plt.plot(est_xy[:, 0], est_xy[:, 1], '-o', color= 'blue', label='Estimated')
    plt.plot(gt_xy[:, 0],  gt_xy[:, 1],  '-x', color='red', label='Ground Truth')

    arrow_scale = 1.0
    head_width = 0.2
    head_length = 0.4
    for i in range(0, N, max(1, N // 10)):
        ex, ey, eyaw = est_xy[i][0], est_xy[i][1], est_yaw[i]
        gx, gy, gyaw = gt_xy[i][0], gt_xy[i][1], gt_yaw[i]
        plt.arrow(ex, ey, arrow_scale * np.cos(eyaw), arrow_scale * np.sin(eyaw),
                  head_width=head_width, head_length=head_length, color='blue', alpha=0.6)
        plt.arrow(gx, gy, arrow_scale * np.cos(gyaw), arrow_scale * np.sin(gyaw),
                  head_width=head_width, head_length=head_length, color='red', alpha=0.6)

    plt.title(f"Epoch {epoch}: Estimated vs Ground Truth Poses. RMSE = {rmse_trans:.3f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/poses_{epoch}')
    else:
        plt.show()    
    plt.close('all')
    

def compute_rmse_and_yaw(graph, gt_poses, delta_x, epoch, plot=False, output_dir=None):
    est_poses = graph.get_estimated_state()
    N = len(est_poses)

    sum_trans_sq = 0.0
    sum_rot_sq = 0.0
    est_xy, gt_xy = [], []
    est_yaw, gt_yaw = [], []

    for i in range(N):
        xi = delta_x[i]
        rot_vec, trans_vec = xi[:3], xi[3:]

        sum_rot_sq += np.sum(rot_vec**2)
        sum_trans_sq += np.sum(trans_vec**2)

        est_pose = est_poses[i]
        gt_pose = gt_poses[i]

        est_xy.append(est_pose[:2, 3])
        gt_xy.append(gt_pose[:2, 3])

        est_yaw.append(np.arctan2(est_pose[1, 0], est_pose[0, 0]))
        gt_yaw.append(np.arctan2(gt_pose[1, 0], gt_pose[0, 0]))

    rmse_rot = np.sqrt(sum_rot_sq / N)
    rmse_trans = np.sqrt(sum_trans_sq / N)
    
    if plot:
        plot_pose_comparison(np.array(est_xy), np.array(gt_xy), np.unwrap(np.array(est_yaw)), np.unwrap(np.array(gt_yaw)), epoch, rmse_trans, output_dir)

    return rmse_trans


def print_2d_graph(graph, gt_poses):
    x = graph.get_estimated_state()
    
    prev_p = np.array(graph.get_estimated_state()[0][:2, 3])
    plt.figure()

    for p in x:
        p = p[:2, 3]
        plt.plot(p[0], p[1], 'ob')
        plt.plot((prev_p[0], p[0]), (prev_p[1], p[1]), '-b', label='estimated')
        
        prev_p = p
    
    plt.plot(gt_poses[:, :2][:, 0], gt_poses[:, :2][:, 1], label='GT', color='red')
    plt.title("2D Pose Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()


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

if __name__ == "__main__":
    model = ResNet1D(
        num_inputs=3,       
        num_outputs=2,         
        block_type=BasicBlock1D,
        group_sizes=[2, 2, 2],   
        base_plane=64,
        output_block=FCOutputModule,  
        kernel_size=3,
        fc_dim=512,             
        in_dim=1,            
        dropout=0.5,
        trans_planes=128
    )
    
    # model.load_state_dict(torch.load("out/model_fgo.pth"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    criterion = nn.MSELoss()

    output_path = './out/'
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    path_to_splines = output_path + 'splines'

    number_of_splines = 10
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)
        
    window_size = 10
    dataset = Spline_2D_Dataset(path_to_splines, window=window_size, enable_noise= not True)

    train_len = 8
    valid_len = 2
    
    print(f'Train dataset length: {train_len}, Validation dataset length: {valid_len}')
    
    train_dataset, valid_dataset = random_split(dataset,[train_len, valid_len])

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    output_path = 'out'
    num_epochs = 100
    
    rmse_errors = []
    chi2_errors = []
    valid_chi2_errors = []
    valid_rmse_errors = []
    
    alpha = 0.0
    dt = window_size / 100
    
    for epoch in range(num_epochs):
        total_chi2 = 0.0
        total_rmse = 0.0
        train_batches = 0.0
        model.train()
        for imu_seq, velocity_seq, pose_seq, w_z_seq in train_dataloader:
            imu_seq = imu_seq.squeeze(0)       # [num_windows, 3, W]
            velocity_seq = velocity_seq.squeeze(0)  # [num_windows, 2]
            pose_seq = pose_seq.squeeze(0)     # [num_windows, 3]
            w_z_seq = w_z_seq.squeeze(0).squeeze(-1)  # [num_windows]

            pred_vel_seq = model(imu_seq) # [num_windows, 2]

            # integrate into poses
            pred_poses = integrate_pred_vel(pred_vel_seq, pose_seq, pose_seq[0], dt=dt)
            if epoch == num_epochs - 1:
                plot_integrated_vel(pred_poses, pose_seq)

            graph = populate_graph(pred_poses, pose_seq)
            graph.build_jacobians()
            graph.solve(mrob.FGraphDiff_LM, maxIters=100) 
            
            N = velocity_seq.shape[0]

            graph.solve(mrob.FGraphDiff_LM, maxIters=100)
            chi2 = graph.chi2()
            
            dL_dz = graph.get_dx_dz()
            dL_dz /= dt

            # visualize_gradient(dL_dz, 'Analytical gradient', 'out', epoch)
           
            gt_poses_se3 = convert_to_se3(pose_seq.detach().cpu().numpy()) 
            delta_x = compute_delta_x(gt_poses_se3, graph.get_estimated_state())
            delta_x_flat = delta_x.reshape(-1)
            # delta_x_flat = delta_x_flat / (np.linalg.norm(delta_x_flat) + 1e-8)
            
            rmse_trans = compute_rmse_and_yaw(graph, gt_poses_se3, delta_x, epoch, plot=True, output_dir='out/poses_train')
            
            mult = (delta_x_flat @ dL_dz)[0:(N * 6)].reshape(-1, 6) #
            mult_tensor = torch.from_numpy(np.array(mult)).float()

            grad_final = -mult_tensor[:, [3, 4]]

            pred_vel_seq.backward(gradient=grad_final.type(torch.float32))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            total_chi2 += chi2
            total_rmse += rmse_trans
            train_batches += 1

            # grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
            # print(f"Grad norm: {grad_norm.item()}")
            
            optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
        chi2_errors.append(total_chi2 / len(train_dataloader))
        rmse_errors.append(rmse_trans / len(train_dataloader))
        
        # Validation loop 
        # model.eval()
        # val_total_chi2 = 0.0
        # val_total_rmse = 0.0
        # val_batches = 0

        # with torch.no_grad():
        #     for imu_seq, velocity_seq, pose_seq, w_z_seq in valid_dataloader:
        #         imu_seq = imu_seq.squeeze(0)       # [num_windows, 3, W]
        #         velocity_seq = velocity_seq.squeeze(0)  # [num_windows, 2]
        #         pose_seq = pose_seq.squeeze(0)     # [num_windows, 3]
        #         w_z_seq = w_z_seq.squeeze(0).squeeze(-1)  # [num_windows]

        #         pred_vel_seq = model(imu_seq)
        #         pred_poses = integrate_pred_vel(pred_vel_seq, pose_seq, pose_seq[0], dt=dt)

        #         graph = populate_graph(pred_poses, pose_seq)
        #         graph.build_jacobians()
        #         graph.solve(mrob.FGraphDiff_LM, maxIters=100)

        #         chi2 = graph.chi2()

        #         gt_poses_se3 = convert_to_se3(pose_seq.detach().cpu().numpy()) 
        #         delta_x = compute_delta_x(gt_poses_se3, graph.get_estimated_state())

        #         rmse_trans = compute_rmse_and_yaw(graph, gt_poses_se3, delta_x, epoch, plot=True, output_dir='out/poses_valid')

        #         val_total_chi2 += chi2
        #         val_total_rmse += rmse_trans
        #         val_batches += 1

        #         valid_chi2_errors.append(val_total_chi2 / val_batches)
        #         valid_rmse_errors.append(val_total_rmse / val_batches)
                
        print(f"Epoch {epoch}, [Training] Chi2: {total_chi2 / train_batches:.5f}, RMSE: {rmse_trans / train_batches:.5f}")
        # print(f"Epoch {epoch}, [Validation] Chi2: {val_total_chi2 / val_batches:.5f}, RMSE: {val_total_rmse / val_batches:.5f}")
    torch.save(model.state_dict(), "out/model.pth")
    plot_losses(chi2_errors, rmse_errors, ('Chi2', 'RMSE'), 'Train Losses', f'{output_path}/train_losses.png')
    # plot_losses(valid_chi2_errors, valid_rmse_errors, ('Chi2', 'RMSE'), 'Valid Losses', f'{output_path}/valid_losses.png')
    
    make_gif_from_figures(f'{output_path}/poses_train', output_path=f'{output_path}/train_graph_evolution.gif', num_epochs=num_epochs)
    # make_gif_from_figures(f'{output_path}/poses_valid', output_path=f'{output_path}/valid_graph_evolution.gif', num_epochs=num_epochs)