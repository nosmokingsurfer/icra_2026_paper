import sys
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime

from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0,str(Path('./external/ronin/source/').resolve()))

from ronin_resnet import get_dataset_from_list, run_test, recon_traj_with_preds, get_dataset
from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
from metric import compute_ate_rte

def build_datasets(args):
    train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    test_dataset = get_dataset_from_list(args.root_dir, args.test_list, args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    print('Number of train samples: {}'.format(len(train_dataset)))
    print('Number of test samples: {}'.format(len(test_dataset)))

    return train_loader, test_loader


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
    I am the supporter of the idea "one model -- one config". So I moved some testues to the config
    In order not to change the init ronin code I delete testues from dictionary once I get them 
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
        raise testueError('Intestid architecture: ', args.arch)
    return network


def train_ronin(args, fc_config):
    output_root_path = Path(args.out_dir)
    current_time = datetime.datetime.now()
    
    output_dir_name = f"{current_time.day}_{current_time.month}_{current_time.hour}_{current_time.minute}"

    output_dir = output_root_path / output_dir_name

    meta_save_dir = output_dir / "meta"
    meta_save_dir.mkdir(parents=True)
    trajectories_save_dir = output_dir / "trajectories"
    trajectories_save_dir.mkdir(parents=True)
    training_checkpoints = output_dir / "training_checkpoints"
    training_checkpoints.mkdir(parents=True)
    test_checkpoints = output_dir / "test_checkpoints"
    test_checkpoints.mkdir(parents=True)

    writer = SummaryWriter(output_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cpu':
        print("_"*50, "\n", "USING CPU FOR TRAININ", "_"*50, "\n")

    train_loader, test_loader = build_datasets(args)
    if args.model_path is not None:
        checkpoints = torch.load(args.model_path)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    else:
        network = get_model(args.arch, fc_config).to(device)
    total_params = network.get_num_params()
    print('Total number of network\'s parameters: ', total_params)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, eps=1e-12)
    # TODO: probably in the future we should move this scheduler parameters to config as well

    train_iteration_tracker, test_iteration_tracker = 1, 1
    best_ate, best_rte = np.inf, np.inf
    try:
        for epoch in range(args.epochs):
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

            model_path = training_checkpoints / f"checkpoint_{epoch}.pt"
            torch.save({'model_state_dict': network.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()}, model_path)
                
            
            if test_iteration_tracker % args.test_step == 0:
                test_outs, test_targets = run_test(network, test_loader, device, etest_mode=True)
                test_losses = np.average((test_outs - test_targets) ** 2, axis=0)
                avg_loss = np.average(test_losses)
                scheduler.step(avg_loss)
                writer.add_scalar('Loss/test', avg_loss, test_iteration_tracker)

                with open(args.test_list) as f:
                    test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
                for data in test_data_list:
                    seq_dataset = get_dataset(args.root_dir, [data], args, mode='test')
                    seq_loader = DataLoader(seq_dataset, batch_size=args.test_batch_size, shuffle=False)
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

                    np.save(trajectories_save_dir / data + '_gsn.npy', np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
                    writer.add_figure(data + '.png', plt.gcf(), global_step=test_iteration_tracker)
                    plt.close('all')
                    

                    losses_seq = np.stack(losses_seq, axis=0)
                    losses_avg = np.mean(losses_seq, axis=1)
                    # Export a csv file

                    with open(output_dir / "losses.csv", 'w') as f:
                        if losses_seq.shape[1] == 2:
                            f.write('seq,vx,vy,avg,ate,rte\n')
                        else:
                            f.write('seq,vx,vy,vz,avg,ate,rte\n')
                        for i in range(losses_seq.shape[0]):
                            f.write('{},'.format(args.test_list[i]))
                            for j in range(losses_seq.shape[1]):
                                f.write('{:.6f},'.format(losses_seq[i][j]))
                            f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

                    if np.mean(ate_all) < best_ate:
                        best_ate = np.mean(ate_all)
                        model_path = test_checkpoints / f"best_ate_%.4f.pt" % best_ate
                        torch.save({'model_state_dict': network.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()}, model_path)
                    if np.mean(rte_all) < best_rte:
                        best_rte = np.mean(rte_all)
                        model_path = test_checkpoints / f"best_rte_%.4f.pt" % best_rte
                        torch.save({'model_state_dict': network.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict()}, model_path)



            test_iteration_tracker += 1

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')




if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/ronin_baseline_config.yaml", help='Path to the config for Ronin')

    starting_args = parser.parse_args()
    config_dict = yaml.safe_load(open(starting_args.config))

    args = argparse.Namespace(**config_dict)

    fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128, "input_channel": 6, "output_channel": 2}
    fc_config['in_dim'] = args.window_size // 32 + 1
    train_ronin(args, fc_config)