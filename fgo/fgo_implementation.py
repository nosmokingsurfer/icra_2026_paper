import torch
import mrob
import numpy as np
from experiments.utils_metrics import compute_rmse_and_yaw

from spline_dataset.spline_dataloader import convert_to_se3

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

def compute_delta_x(gt_poses, estimated_poses):
    gt_poses_se3 = [mrob.SE3(gt_poses[i]) for i in range(len(gt_poses))]
    estimated_poses_se3 = [mrob.SE3(estimated_poses[j]) for j in range(len(estimated_poses))]
    
    result_Ln = [(gt_poses_se3[k] * estimated_poses_se3[k].inv()).Ln() for k in range(len(gt_poses_se3))]
    return np.array(result_Ln)

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
