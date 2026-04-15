import sys
sys.path.insert(0,'.')
import os
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import pytorch_lightning as L
from torch.multiprocessing import Pool
from typing import OrderedDict

from gsdc_dataset.lit_training import LIT_GSDC_datamodule, LIT_SimpleVDRModel
from gsdc_dataset.utils import get_tasks_for_dataset

import torch.utils.tensorboard


if __name__ == "__main__":
    # checkpoint_idx = "version_0"

    checkpoint_idx = "version_777"

    if not os.path.exists(f"./out_vdr/{checkpoint_idx}"):
        os.makedirs(f"./out_vdr/{checkpoint_idx}",exist_ok=True)


    hparams = yaml.safe_load(open(f"./lightning_logs/{checkpoint_idx}/hparams.yaml",'r'))
    dataloader_params=hparams['dataloader_params']

    model = LIT_SimpleVDRModel()

    try:
        model.load_from_checkpoint(f"./lightning_logs/{checkpoint_idx}/checkpoints/last.ckpt")
    except:
        # state_dict = torch.load(open(f"./lightning_logs/{checkpoint_idx}/checkpoints/last.ckpt",'rb'),map_location=model.device)['state_dict']
        state_dict = torch.load(open(f"./lightning_logs/{checkpoint_idx}/checkpoints/epoch=91-val_loss=0.00000-train_loss=2.51320.ckpt",'rb'),map_location=model.device)['state_dict']


        new_state = OrderedDict()
        for k,v in state_dict.items():
            new_state[k.replace('model.model','model')] = v

        model.load_state_dict(new_state)



    data_path = "./data/smartphone-decimeter-2022/"
    tasks = get_tasks_for_dataset()

    for task in tasks:
        task['checkpoint_idx'] = checkpoint_idx

    datamodule = LIT_GSDC_datamodule(tasks, data_path, dataloader_params)

    trainer = L.Trainer(
        # strategy='ddp_find_unused_parameters_true',
        accelerator='auto' if torch.cuda.is_available() else 'cpu',
        devices=[0] if torch.cuda.is_available() else None,
        callbacks = []
    )

    # trainer.test(model, datamodule)

    if not os.path.exists(f'./out_vdr/{checkpoint_idx}/'):
        os.makedirs(f'./out_vdr/{checkpoint_idx}/',exist_ok=True)

    for task in tqdm(tasks):
        pred_path = f"./out_vdr/{checkpoint_idx}/{task['sample_id']}_predictions.csv"
        preds = pd.read_csv(pred_path)

        rotated_data_path = f"./out_vdr/train/{task['sample_id']}_rotated_combined_data.pickle"
        rotated_data = pd.read_pickle(rotated_data_path)

        fig,ax = plt.subplots(3,1)
        fig.suptitle(task['sample_id'])

        ax[0].plot(preds['pred_s_vx'],label='pred_vx')
        ax[0].plot(preds['gt_s_vx'],label='gt_vx')
        ax[0].grid()
        ax[0].legend()

        ax[1].plot(preds['pred_s_vy'],label='pred_vy')
        ax[1].plot(preds['gt_s_vy'],label='gt_vy')
        ax[1].grid()
        ax[1].legend()

        ax[2].plot(np.linalg.norm(preds[['pred_s_vx','pred_s_vy']].values,axis=1),label='pred_abs_v')
        ax[2].plot(np.linalg.norm(preds[['gt_s_vx','gt_s_vy']].values,axis=1),label='gt_abs_v')
        ax[2].grid()
        ax[2].legend()

        plt.savefig(f'./out_vdr/{checkpoint_idx}/{task["sample_id"]}_predicted_velocity.png')
        plt.close('all')

        pass
        # print(rotated_data.columns)

        