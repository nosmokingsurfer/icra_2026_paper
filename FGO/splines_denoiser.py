import mrob
import numpy as np
np.set_printoptions(precision=4,linewidth=180)
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mrob_num_diff.graph_generator import ToRoContainer, compare_gradients
from mrob_num_diff.num_diff_3d import numerical_diff2_3d, numerical_diff2_3d_sparse
from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset, convert_to_se3


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
    

class TrivialPoseDenoiser(nn.Module):
    def __init__(self):
        super(TrivialPoseDenoiser, self).__init__()
        self.translation = nn.Linear(2, 2)
        
    def forward(self, x):
        denoised_translation = self.translation(x[:, :2])
        return torch.cat([denoised_translation, x[:, 2:]], dim=1)
    

def compute_delta_x(gt_poses, estimated_poses):
    gt_poses_se3 = [mrob.SE3(gt_poses[i]) for i in range(len(gt_poses))]
    estimated_poses_se3 = [mrob.SE3(estimated_poses[j]) for j in range(len(estimated_poses))]
    
    result_Ln = [(gt_poses_se3[k] * estimated_poses_se3[k].inv()).Ln() for k in range(len(gt_poses_se3))]
    return np.array(result_Ln)


def populate_graph(pred_poses, gt_poses):
    graph = mrob.FGraph()
    toro_container = ToRoContainer()

    W_odo = torch.eye(6) * 10.0
    W_gps = torch.eye(6) * 50.0

    node_ids = []
    T_nodes = []
    # Initialize nodes using predicted poses (world frame)
    for i in range(len(pred_poses)):
        T_i = mrob.SE3(convert_to_se3(pred_poses[i].detach().cpu().numpy()))
        T_nodes.append(T_i)
        node_id = graph.add_node_pose_3d(T_i)
        node_ids.append(node_id)
        toro_container.add_node_pose_3d(node_id, T_i.Ln())

    # Odometry factors from predicted poses
    for i in range(len(node_ids) - 1):
        T_pred_i   = T_nodes[i]
        T_pred_ip1 = T_nodes[i + 1]
        T_odo = T_pred_i.inv() * T_pred_ip1
        graph.add_factor_2poses_3d(T_odo, node_ids[i], node_ids[i + 1], W_odo.numpy())
        toro_container.add_factor_2poses_3d(node_ids[i], node_ids[i + 1], T_odo.Ln(), W_odo.numpy())
    
    # GPS factors from GT poses
    for i in range(len(node_ids)):
        T_gps = mrob.SE3(convert_to_se3(gt_poses[i].detach().cpu().numpy()))
        graph.add_factor_1pose_3d(T_gps, node_ids[i], W_gps.numpy())
        toro_container.add_factor_1pose_3d(node_ids[i], T_gps.Ln(), W_gps.numpy())
    
    return graph, toro_container.get_lines()



def plot_pose_comparison(est_xy, gt_xy, est_yaw, gt_yaw, epoch, rmse_trans, output_dir=None):
    N = len(est_xy)
    plt.figure(figsize=(10, 8))
    plt.plot(est_xy[:, 0], est_xy[:, 1], '-o', color= 'blue', label='Estimated')
    plt.plot(gt_xy[:, 0],  gt_xy[:, 1],  '-x', color='red', label='Ground Truth')

    arrow_scale = 0.1
    head_width = 0.02
    head_length = 0.04
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


def make_gif_from_figures(folder_path, output_path="graph_evolution.gif", duration=0.3):
    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.startswith("poses_") and f.endswith(".png")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    images = [imageio.imread(os.path.join(folder_path, fname)) for fname in file_list]
    imageio.mimsave(output_path, images, duration=duration)
    print(f"GIF saved to {output_path}")


if __name__ == "__main__":
    model = TrivialPoseDenoiser()
    # model.load_state_dict(torch.load("out/model.pth"))
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    path_to_splines = "./out/splines_train"
    
    if not os.path.exists(path_to_splines):
        number_of_splines = 1
        number_of_control_nodes = 10
        n_pts_spline_segment = 100
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, n_pts_spline_segment)
    
    train_dataset = Spline_2D_Dataset(path_to_splines)[0]
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)
    
    valid_dataset = Spline_2D_Dataset(path_to_splines)[1]
    valid_dataloader = DataLoader(valid_dataset, batch_size=100, shuffle=False)
    
    output_path = 'out'
    num_epochs = 100
    
    rmse_errors = []
    chi2_errors = []
    
    rmse_errors_valid = []
    chi2_errors_valid = []
    
    for epoch in tqdm((range(num_epochs))):
        model.train()
        for noisy_poses, gt_poses in train_dataloader:
            model.train()
            pred_poses = model(noisy_poses)
            
            N = gt_poses.shape[0]

            graph, toro_lines = populate_graph(pred_poses, gt_poses)
            graph.solve(mrob.LM)
            chi2 = graph.chi2()
            toro_file = os.path.join(output_path, 'toro_file.txt')
            with open(toro_file,'w') as f:
                f.writelines(toro_lines)
                f.close()

            dL_dz = numerical_diff2_3d_sparse(toro_file, dx=1e-5, dz=1e-4, parallel=False)
            chi2_errors.append(chi2)
           
            gt_poses_se3 = convert_to_se3(gt_poses.detach().cpu().numpy()) 
            delta_x = compute_delta_x(gt_poses_se3, graph.get_estimated_state())
            delta_x_flat = delta_x.reshape(-1)
            
            rmse_trans = compute_rmse_and_yaw(graph, gt_poses_se3, delta_x, epoch, plot=True, output_dir=f'{output_path}/poses_train')
            rmse_errors.append(rmse_trans)
            
            mult = (delta_x_flat @ dL_dz)[0:(N * 6)].reshape(-1, 6)
            mult_tensor = torch.from_numpy(np.array(mult)).float()

            grad_final = mult_tensor[:, [3, 4, 2]] 

            assert pred_poses.requires_grad
            
            optimizer.zero_grad()
            pred_poses.backward(gradient=grad_final.type(torch.float32))
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            for noisy_poses_valid, gt_poses_valid in valid_dataloader:
                pred_poses_valid = model(noisy_poses)
                graph_valid, _ = populate_graph(pred_poses_valid, gt_poses_valid)
                graph_valid.solve(mrob.LM)
                chi2_valid = graph_valid.chi2()
                delta_x_valid = compute_delta_x(convert_to_se3(gt_poses_valid.numpy()), graph_valid.get_estimated_state())
                rmse_trans_valid = compute_rmse_and_yaw(graph_valid, convert_to_se3(gt_poses_valid.numpy()), delta_x_valid, epoch, plot=True, output_dir='out/poses_valid')
                rmse_errors_valid.append(rmse_trans_valid)
                chi2_errors_valid.append(chi2_valid)
            
        print(f"Epoch {epoch}")
        print(f"Chi2 on training: {chi2}, RMSE on training: {rmse_trans}")
        print(f"Chi2 on validation: {chi2_valid}, RMSE on validation: {rmse_trans_valid}") 

    plot_losses(chi2_errors, rmse_errors, ('Chi2', 'RMSE'), 'Train Losses', f'{output_path}/train_losses.png')
    make_gif_from_figures(f'{output_path}/poses_train', output_path=f'{output_path}/train_graph_evolution.gif')
    
    plot_losses(chi2_errors_valid, rmse_errors_valid, ('Chi2', 'RMSE'), 'Valid Losses', f'{output_path}/valid_losses.png')
    make_gif_from_figures(f'{output_path}/poses_valid', output_path=f'{output_path}/valid_graph_evolution.gif')