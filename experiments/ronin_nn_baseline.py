import sys
from pathlib import Path
from os import path as osp
# TODO: remove os.path usage, since we are already using pathlib
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0,str(Path('./external/ronin/source/').resolve()))

from ronin_resnet import get_dataset_from_list, run_test, recon_traj_with_preds, get_dataset
from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
from metric import compute_ate_rte

def build_datasets(args):
    train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))

    return train_dataset, train_loader, val_dataset, val_loader


def get_model(arch, fc_config):
    """
    Almost duplicate of the function from ronin, but I considered it is not safe to use from there
    So I drought it here
    """
    input_channel = fc_config["input_channel"]
    output_channel = fc_config["output_channel"]

    del fc_config["input_channel"]
    del fc_config["output_channel"]
    """
    I am the supporter of the idea "one model -- one config". So I moved some values to the config
    In order not to change the init ronin code I delete values from dictionary once I get them 
    """
    if arch == 'resnet18':
        network = ResNet1D(input_channel, output_channel, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **fc_config)
    elif arch == 'resnet50':
        # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
        fc_config['fc_dim'] = 1024
        network = ResNet1D(input_channel, output_channel, BasicBlock1D, [3, 4, 6, 3],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **fc_config)
    elif arch == 'resnet101':
        fc_config['fc_dim'] = 1024
        network = ResNet1D(input_channel, output_channel, BasicBlock1D, [3, 4, 23, 3],
                           base_plane=64, output_block=FCOutputModule, **fc_config)
    else:
        raise ValueError('Invalid architecture: ', args.arch)
    return network


def train_ronin(args, fc_config):

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cpu':
        print("_"*50, "\n", "USING CPU FOR TRAININ", "_"*50, "\n")

    train_dataset, train_loader, val_dataset, val_loader = build_datasets(args)

    network = get_model(args.arch, fc_config).to(device)
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, eps=1e-12)
    train_losses_all, val_losses_all = [], []

    init_train_targ, init_train_pred = run_test(network, train_loader, device, eval_mode=False)

    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    best_val_loss = np.inf

    writer = SummaryWriter(args.out_dir) 
    train_iteration_tracker, val_iteration_tracker = 1, 1

    try:
        for _ in range(args.epochs):
            print("val_iteration_tracker:  ", val_iteration_tracker)
            network.train()
            for feat, targ, _, _ in train_loader:  # feat targ and ???????? why do we even need to more args if we do not us it?
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                pred = network(feat)
                train_loss = criterion(pred, targ)
                train_loss = torch.mean(train_loss)
                train_loss.backward()
                optimizer.step()
                writer.add_scalar('Loss/train', train_loss.detach().cpu(), train_iteration_tracker)
                train_iteration_tracker += 1

            if val_iteration_tracker % args.validation_step == 0:
                if val_loader is not None:
                    val_outs, val_targets = run_test(network, val_loader, device, eval_mode=True)
                    val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                    avg_loss = np.average(val_losses)
                    scheduler.step(avg_loss)
                    writer.add_scalar('Loss/val', avg_loss, val_iteration_tracker)
            val_iteration_tracker += 1


            if val_iteration_tracker % args.display_trajectories_step == 0:
                with open(args.val_list) as f:
                    test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
                for data in test_data_list:
                    print(args.root_dir, data)
                    seq_dataset = get_dataset(args.root_dir, [data], args, mode='test')
                    seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
                    ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int32)
                    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
                    traj_lens = []
                    pred_per_min = 200 * 60

                    targets, preds = run_test(network, seq_loader, device, True)
                    losses = np.mean((targets - preds) ** 2, axis=0)
                    preds_seq.append(preds)
                    targets_seq.append(targets)
                    losses_seq.append(losses)

                    pos_pred = recon_traj_with_preds(seq_dataset, preds)[:, :2]
                    pos_gt = seq_dataset.gt_pos[0][:, :2]

                    traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
                    ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
                    ate_all.append(ate)
                    rte_all.append(rte)
                    pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

                    # Plot figures
                    kp = preds.shape[1]
                    if kp == 2:
                        targ_names = ['vx', 'vy']
                    elif kp == 3:
                        targ_names = ['vx', 'vy', 'vz']

                    plt.figure('{}'.format(data), figsize=(16, 9))
                    plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
                    plt.plot(pos_pred[:, 0], pos_pred[:, 1])
                    plt.plot(pos_gt[:, 0], pos_gt[:, 1])
                    plt.title(data)
                    plt.axis('equal')
                    plt.legend(['Predicted', 'Ground truth'])
                    plt.subplot2grid((kp, 2), (kp - 1, 0))
                    plt.plot(pos_cum_error)
                    plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
                    for i in range(kp):
                        plt.subplot2grid((kp, 2), (i, 1))
                        plt.plot(ind, preds[:, i])
                        plt.plot(ind, targets[:, i])
                        plt.legend(['Predicted', 'Ground truth'])
                        plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
                    plt.tight_layout()

                    if args.show_plot:
                        plt.show()

                    print(args.out_dir, osp.isdir(args.out_dir))
                    if args.out_dir is not None and osp.isdir(args.out_dir):
                        np.save(osp.join("./", "data", data + '_gsn.npy'),
                                np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
                        plt.savefig(osp.join("./data", data + '_gsn.png'))
                    writer.add_figure(data + '.png', plt.gcf(), global_step=val_iteration_tracker)
                    plt.close('all')
                    

                    losses_seq = np.stack(losses_seq, axis=0)
                    losses_avg = np.mean(losses_seq, axis=1)
                    # Export a csv file
                    if args.out_dir is not None and osp.isdir(args.out_dir):
                        with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
                            if losses_seq.shape[1] == 2:
                                f.write('seq,vx,vy,avg,ate,rte\n')
                            else:
                                f.write('seq,vx,vy,vz,avg,ate,rte\n')
                            for i in range(losses_seq.shape[0]):
                                f.write('{},'.format(args.val_list[i]))
                                for j in range(losses_seq.shape[1]):
                                    f.write('{:.6f},'.format(losses_seq[i][j]))
                                f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))



    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    parser.add_argument('--validation_step', type=int, default=1, help='How often, in terms of epochs, run validation')
    parser.add_argument('--display_trajectories_step', type=int, default=1, help='How often, in terms of epochs, dusplay trajectories')

    args = parser.parse_args()


    fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128, "input_channel": 6, "output_channel": 2}
    fc_config['in_dim'] = args.window_size // 32 + 1
    print(type(args))

    train_ronin(args, fc_config)