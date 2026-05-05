import torch
import pytorch_lightning as L
from  pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, BatchSizeFinder

import torch.nn.functional as F
from  torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.multiprocessing import Pool
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

import sys
sys.path.insert(0,'.')
from gsdc_dataset.simple_vdr_model import SimpleVDRmodel
from gsdc_dataset.gsdc_dataloader import GSDC_dataset, generate_combined_data, rotate_data, zero_padding_collate
from gsdc_dataset.utils import get_tasks_for_dataset

import torch.utils.tensorboard

class LIT_SimpleVDRModel(L.LightningModule):
    def __init__(self, dataloader_params):
        super(LIT_SimpleVDRModel,self).__init__()
        self.model = SimpleVDRmodel()
        self.dataloader_params = dataloader_params
        self.hparams.update(
            {"architecture" : "simple_vdr_model"}
        )
        self.save_hyperparameters()


    def forward(self,X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.dataloader_params['learning_rate'])

        lr_scheduler = ReduceLROnPlateau(optimizer,factor=0.97, patience=3)
        return {
            "optimizer" : optimizer,
            "lr_scheduler" : {
                "scheduler" : lr_scheduler,
                "interval" : "epoch",
                "monitor" : "train_loss"
            }
        }


    def training_step(self, train_batch, batch_idx):
        acc = train_batch['acc']
        gyro = train_batch['gyro']
        gt_vel = train_batch['gt_velocity']
        gt_traj = train_batch['gt_traj']

        imu = torch.concat((acc,gyro),dim=-1).swapaxes(-1,-2)

        pred = self.model(imu)

        window = torch.ones((2,1,8))/8
        window = window.to(gt_vel.device)
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)

        loss = torch.nn.functional.mse_loss(pred,gt_vel)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        acc = val_batch['acc']
        gyro = val_batch['gyro']
        gt_vel = val_batch['gt_velocity']
        gt_traj = val_batch['gt_traj']

        imu = torch.concat((acc,gyro),dim=-1).swapaxes(-1,-2)

        pred = self.model(imu)

        window = torch.ones((2,1,8))/8
        window = window.to(gt_vel.device)
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)

        loss = torch.nn.functional.mse_loss(pred,gt_vel)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, test_batch, test_idx):
        lengths = test_batch['lengths'].detach().numpy()
        acc = test_batch['acc']
        gyro = test_batch['gyro']
        gt_vel = test_batch['gt_velocity']
        gt_traj = test_batch['gt_traj']
        task = test_batch['task']

        sample_id = [t['sample_id'] for t in task]
        checkpoit_idx = [t['checkpoint_idx'] for t in task]

        imu = torch.concat((acc,gyro),dim=-1).swapaxes(-1,-2)

        pred = self.model(imu)

        window = torch.ones((2,1,8))/8
        window = window.to(gt_vel.device)
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8,groups=2,).swapaxes(-1,-2)

        lengths = lengths//8

        minlength = min(pred.shape[1],gt_vel.shape[1])
        pred = pred[:,:minlength]
        gt_vel = gt_vel[:,:minlength]

        loss = torch.nn.functional.mse_loss(pred,gt_vel)
        self.log('test_loss', loss, on_epoch=True,batch_size=test_batch['acc'].shape[0])

        for b in range(test_batch['acc'].shape[0]):
            result = {}
            pred_path = f"./out_vdr/{checkpoit_idx[b]}/{sample_id[b]}_predictions.csv"

            result = pd.DataFrame(np.concatenate((pred[b].detach().numpy(),gt_vel[b].detach().numpy()),axis=-1), columns = ['pred_s_vx','pred_s_vy','gt_s_vx','gt_s_vy'])
            result = result[:lengths[b]]
            result.to_csv(pred_path, index=False)
        return loss

class LIT_GSDC_datamodule(L.LightningDataModule):
    def __init__(self, tasks, data_path, dataloader_params):
        super().__init__()
        self.tasks = tasks
        self.data_path = data_path
        self.dataloader_params = dataloader_params
        self.hparams.update(dataloader_params)
        self.save_hyperparameters()

        self.train_tasks = None
        self.val_tasks = None

    def setup(self, stage):
        super().setup(stage)
        train_ratio = self.dataloader_params['train_ratio']

        num_train = int(len(self.tasks)*train_ratio)

        import random
        shuffled_tasks = self.tasks.copy()
        random.shuffle(shuffled_tasks)

        self.train_tasks = shuffled_tasks[:num_train]
        self.val_tasks = shuffled_tasks[num_train:]


    def train_dataloader(self):
        dataset = GSDC_dataset('train', self.train_tasks, self.data_path, **self.dataloader_params)
        return DataLoader(dataset,self.dataloader_params['batch_size'], shuffle=True, drop_last=True,num_workers=8)

    def val_dataloader(self):
        dataset = GSDC_dataset('val', self.val_tasks, self.data_path, **self.dataloader_params)
        return DataLoader(dataset,self.dataloader_params['batch_size'], shuffle=False, drop_last=True, num_workers=8)

    def test_dataloader(self):
        dataset = GSDC_dataset('test', self.tasks, self.data_path, **self.dataloader_params)
        return DataLoader(dataset,batch_size=2, shuffle=False, drop_last=True, num_workers=2, collate_fn=zero_padding_collate)


if __name__ == "__main__":
    # batch_size_finder = BatchSizeFinder() # TODO

    dataloader_params = {
        "window" : 8*125*10,
        "step" : 1000,
        "frequency": 50,
        "batch_size" : 32,
        "train_ratio" : 0.7,
        "learning_rate" : 1e-3
    }

    data_path = "./data/smartphone-decimeter-2022/"
    tasks = get_tasks_for_dataset(data_path)
    tasks = tasks[:2]

    print('Preprocessing tasks...')
    with Pool(6) as p:
        # generating combined_data
        res = [p.apply_async(generate_combined_data,args=(t,dataloader_params['frequency'])) for t in tasks]
        for r in tqdm(res):
            r.get()

    with Pool(6) as p:
        # rotating data
        res = [p.apply_async(rotate_data,args=(t,dataloader_params['frequency'])) for t in tasks]
        for idx,r in tqdm(enumerate(res),total=len(tasks)):
            r.get()


    model = LIT_SimpleVDRModel(dataloader_params)
    gsdc_datamodule = LIT_GSDC_datamodule(tasks, data_path, dataloader_params)
    gsdc_datamodule.setup('fit')


    lr_monitor = LearningRateMonitor('epoch')

    checkpoint_monitor = ModelCheckpoint(
        filename='{epoch}-{val_loss:.5f}-{train_loss:.5f}',
        save_top_k=5,
        save_last=True,
        monitor='train_loss'
        )


    trainer = L.Trainer(
        # strategy='ddp_find_unused_parameters_true',
        accelerator='auto' if torch.cuda.is_available() else 'cpu',
        devices=[0] if torch.cuda.is_available() else None,
        max_epochs=1000,
        callbacks = [
            lr_monitor,
            checkpoint_monitor,
            ]
    )

    trainer.fit(model, gsdc_datamodule)
