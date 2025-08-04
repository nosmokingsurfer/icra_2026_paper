import matplotlib.pyplot as plt
import os

from pathlib import Path
import sys
sys.path.insert(0,str(Path('./external/ronin/source/').resolve()))

import numpy as np

from metric import compute_ate_rte

def compute_rmse_and_yaw(graph, gt_poses, delta_x, plot=False, output_dir=None):
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

    return rmse_trans