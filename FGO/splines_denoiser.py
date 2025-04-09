import mrob
import numpy as np
np.set_printoptions(precision=4,linewidth=180)
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mrob_num_diff.graph_generator import ToRoContainer
from mrob_num_diff.num_diff import compare_gradients, visualize_gradient
from mrob_num_diff.num_diff_3d import numerical_diff2_3d, numerical_diff1_3d
from spline_dataset.spline_generation import generate_batch_of_splines
from spline_dataset.spline_dataloader import Spline_2D_Dataset, convert_to_se3


def convert_to_se3_torch(gt_poses):
    num_poses = gt_poses.shape[0]
    se3_matrices = []
    for i in range(num_poses):
        x, y, cos_theta, sin_theta = gt_poses[i]
        theta = torch.atan2(sin_theta, cos_theta)
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta),  torch.cos(theta), 0.0],
            [0.0,              0.0,             1.0]
        ], device=gt_poses.device, dtype=gt_poses.dtype)
        T = torch.eye(4, device=gt_poses.device, dtype=gt_poses.dtype)
        T[:3, :3] = R
        T[:3, 3] = torch.tensor([x, y, 0.0], device=gt_poses.device, dtype=gt_poses.dtype)
        se3_matrices.append(T)
    return torch.stack(se3_matrices, dim=0)
   
   
def plot_losses(losses, title):
    plt.figure(figsize=(12, 6))
    plt.plot(losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.title(f'{title} per Epoch', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()   
    
    
class TrivialPoseDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)  # input: noisy [x, y, cos, sin] → output: cleaned

    def forward(self, x):
        return self.linear(x)

class TrivialPoseDenoiser(nn.Module):
    def __init__(self):
        super(TrivialPoseDenoiser, self).__init__()
        self.translation = nn.Linear(2, 2)
        
    def forward(self, x):
        denoised_translation = self.translation(x[:, :2])
        return torch.cat([denoised_translation, x[:, 2:]], dim=1)

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
    

def compute_delta_x(gt_poses, estimated_poses):
    gt_poses_se3 = [mrob.SE3(gt_poses[i]) for i in range(len(gt_poses))]
    estimated_poses_se3 = [mrob.SE3(estimated_poses[j]) for j in range(len(estimated_poses))]
    
    result_Ln = [(gt_poses_se3[k] * estimated_poses_se3[k].inv()).Ln() for k in range(len(gt_poses_se3))]
    return np.array(result_Ln)


def populate_graph(pred_poses, gt_poses):
    graph = mrob.FGraph()
    toro_container = ToRoContainer()

    W_odo = torch.eye(6) * 10.0
    W_gps = torch.eye(6) * 100.0

    node_ids = []
    T_nodes = []
    # Initialize nodes using predicted poses (world frame)
    for i in range(len(pred_poses)):
        T_i = mrob.SE3(convert_to_se3_torch(pred_poses[i].unsqueeze(0))[0])
        T_nodes.append(T_i)
        node_id = graph.add_node_pose_3d(T_i)
        node_ids.append(node_id)
        toro_container.add_node_pose_3d(node_id, T_i.Ln())

    # Odometry factors 
    for i in range(len(node_ids) - 1):
        T_pred_i   = T_nodes[i]
        T_pred_ip1 = T_nodes[i + 1]
        T_odo = T_pred_i.inv() * T_pred_ip1
        graph.add_factor_2poses_3d(T_odo, node_ids[i], node_ids[i + 1], W_odo.numpy())
        toro_container.add_factor_2poses_3d(node_ids[i], node_ids[i + 1], T_odo.Ln(), W_odo.numpy())
    
    # GPS factors
    for i in range(len(node_ids)):
        T_gps = mrob.SE3(convert_to_se3_torch(gt_poses[i].unsqueeze(0))[0])
        graph.add_factor_1pose_3d(T_gps, node_ids[i], W_gps.numpy())
        toro_container.add_factor_1pose_3d(node_ids[i], T_gps.Ln(), W_gps.numpy())
    
    return graph, toro_container.get_lines()


def compare_estimated_poses(graph, gt_poses, noisy_poses, delta_x, epoch):
    est_poses = graph.get_estimated_state()
    N = len(est_poses)

    sum_trans_sq = 0.0
    sum_rot_sq   = 0.0

    est_xy = []
    gt_xy  = []
    noisy_xy = []

    for i in range(N):
        # delta = mrob.SE3(est_poses[i]) * mrob.SE3(gt_poses[i]).inv()
        # xi = delta.Ln()
        xi = delta_x[i]
        
        rot_vec = xi[:3]
        trans_vec = xi[3:]
        
        sum_rot_sq   += np.sum(rot_vec**2)
        sum_trans_sq += np.sum(trans_vec**2)

        e_x = est_poses[i][0, 3]
        e_y = est_poses[i][1, 3]
        est_xy.append([e_x, e_y])

        g_x = gt_poses[i][0, 3]
        g_y = gt_poses[i][1, 3]
        gt_xy.append([g_x, g_y])
        
        n_x = noisy_poses[i][0, 3]
        n_y = noisy_poses[i][1, 3]
        noisy_xy.append([n_x, n_y])

    rmse_rot   = np.sqrt(sum_rot_sq   / N)  # rotation norm
    rmse_trans = np.sqrt(sum_trans_sq / N)  # translation norm

    # print(f"RMSE (rotation)    : {rmse_rot}")
    print(f"RMSE (translation) : {rmse_trans}")

    est_xy = np.array(est_xy)
    gt_xy  = np.array(gt_xy)
    noisy_xy = np.array(noisy_xy)

    plt.figure()
    plt.plot(est_xy[:, 0], est_xy[:, 1], '-o', label='Estimated')
    plt.plot(gt_xy[:, 0],  gt_xy[:, 1],  '-x', label='Ground Truth')
    # plt.plot(noisy_xy[:, 0], noisy_xy[:, 1], '--', label='Noisy')
    plt.title("Pose Comparison (2D Projection of SE(3))")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    os.makedirs('out/poses', exist_ok=True)
    plt.savefig(f'out/poses/poses_{epoch}')
    plt.close('all')



if __name__ == "__main__":
    model = TrivialPoseDenoiser()
    # model.load_state_dict(torch.load("out/model.pth"))
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.001)
    criterion = nn.MSELoss()
    path_to_splines = "./out/spline_dataset"
    number_of_splines = 10
    
    if not os.path.exists(path_to_splines):
        number_of_control_nodes = 10
        generate_batch_of_splines(path_to_splines, number_of_splines, number_of_control_nodes, 100)
    
    dataset = Spline_2D_Dataset(path_to_splines, enable_noise=False)[0]
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    output_path = 'out'
    num_epochs = 100
    
    losses = []
    chi2_errors = []
    for epoch in tqdm((range(num_epochs))):
        epoch_loss = 0
        model.train()
        for noisy_poses, gt_poses in dataloader: 
            model.train()
            pred_poses = model(noisy_poses)

            graph, toro_lines = populate_graph(pred_poses, gt_poses)
            graph.solve(mrob.LM)
            print()
            print(f"Epoch {epoch}: Chi-squared error: {graph.chi2()}")
            toro_file = os.path.join(output_path, 'toro_file.txt')
            with open(toro_file,'w') as f:
                f.writelines(toro_lines)
                f.close()

            chi2_dx_dz = numerical_diff2_3d(toro_file, dx=1e-5, dz=1e-4)
            chi2_errors.append(graph.chi2())
           
            gt_poses_se3 = convert_to_se3_torch(gt_poses) 
            delta_x = compute_delta_x(gt_poses_se3, graph.get_estimated_state())
            
            noisy_poses_se3 = convert_to_se3_torch(noisy_poses) 
            compare_estimated_poses(graph, gt_poses_se3, noisy_poses_se3, delta_x, epoch)
            
            N = chi2_dx_dz.shape[0]
            mult = (delta_x.reshape(-1) @ chi2_dx_dz)[:, 0:N].reshape(-1, 6)
            mult_tensor = torch.from_numpy(np.array(mult)).float()
            
            grad_trans = mult_tensor[:, 3:5]    # for x y
            grad_yaw = mult_tensor[:, 2]        # for yaw
            
            pred_angle_tensor = torch.atan2(pred_poses[:, 3], pred_poses[:, 2])
            grad_angle = grad_yaw.unsqueeze(1) * torch.stack([-torch.sin(pred_angle_tensor), torch.cos(pred_angle_tensor)], dim=1)

            grad_final = torch.cat([grad_trans, grad_angle], dim=1)
            # print(grad_final)

            optimizer.zero_grad()
            pred_poses.backward(gradient=grad_final.type(torch.float32))
            optimizer.step()
            
        scheduler.step()
    
    plot_losses(chi2_errors, 'Chi2')