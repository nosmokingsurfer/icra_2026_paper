import torch
import pytorch_lightning as L
from  pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, BatchSizeFinder

import torch.functional as F
from  torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path

import theseus as th

import sys
sys.path.insert(0,'.')
from gsdc_dataset.th_simple_model import FGO_SimpleVDRModel
from gsdc_dataset.gsdc_dataloader import GSDC_dataset, generate_combined_data, rotate_data
from gsdc_dataset.lit_training import LIT_GSDC_datamodule
from gsdc_dataset.utils import get_tasks_for_dataset


import torch.utils.tensorboard

class LIT_TH_FGO_SimpleVDRModel(L.LightningModule):
    def __init__(self,dataloader_params):
        super(LIT_TH_FGO_SimpleVDRModel,self).__init__()
        self.dataloader_params = dataloader_params
        self.model = FGO_SimpleVDRModel(**dataloader_params)
        self.hparams.update(dataloader_params)
        self.hparams.update({'architecture' : "th_fgo_simple_vdr_model"})
        self.save_hyperparameters()


    def forward(self,X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        lr_scheduler = ReduceLROnPlateau(optimizer,factor=0.95, patience=3)
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
        yaw_angle = train_batch['yaw_angle']

        out, status = self.model(train_batch)

        # computing loss
        window = torch.ones((2,1,8))/8
        window = window.to(gt_traj.device)
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)
        gt_traj = torch.conv1d(gt_traj.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)

        window=torch.ones((1,1,8))/8
        window = window.to(gt_traj.device)

        yaw_angle=torch.conv1d(yaw_angle.unsqueeze(-2),window, stride=8).swapaxes(-1,-2).squeeze()

        gt_traj_se2 = torch.concat((gt_traj,yaw_angle.unsqueeze(-1)),dim=-1)
        
        theseus_input = {}
        for i in range(self.model.N):
            theseus_input[f'gt_pose_{i}'] = th.SE2(x_y_theta=gt_traj_se2[:,i,:].detach()).tensor

        losses = []
        for i in range(self.model.N):
            p = th.SE2(tensor=out[f'pose_{i}'])
            gt = th.SE2(tensor=theseus_input[f'gt_pose_{i}'])
            
            losses.append(torch.linalg.norm(p.local(gt),dim=-1))

        losses = torch.stack(losses, dim=-1)

        loss = torch.mean(losses)

        self.log('train_loss', loss, on_epoch=True,prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        acc = val_batch['acc']
        gyro = val_batch['gyro']
        gt_vel = val_batch['gt_velocity']
        gt_traj = val_batch['gt_traj']
        yaw_angle = val_batch['yaw_angle']

        out, status = self.model(val_batch)

        # computing loss
        window = torch.ones((2,1,8))/8
        window = window.to(gt_traj.device)
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)
        gt_traj = torch.conv1d(gt_traj.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)

        window=torch.ones((1,1,8))/8
        window = window.to(gt_traj.device)

        yaw_angle=torch.conv1d(yaw_angle.unsqueeze(-2),window, stride=8).swapaxes(-1,-2).squeeze()

        gt_traj_se2 = torch.concat((gt_traj,yaw_angle.unsqueeze(-1)),dim=-1)
        
        theseus_input = {}
        for i in range(self.model.N):
            theseus_input[f'gt_pose_{i}'] = th.SE2(x_y_theta=gt_traj_se2[:,i,:].detach()).tensor

        losses = []
        for i in range(self.model.N):
            p = th.SE2(tensor=out[f'pose_{i}'])
            gt = th.SE2(tensor=theseus_input[f'gt_pose_{i}'])
            
            losses.append(torch.linalg.norm(p.local(gt),dim=-1))

        losses = torch.stack(losses, dim=-1)

        loss = torch.mean(losses)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss


if __name__ == "__main__":

    # batch_size_finder = BatchSizeFinder() # TODO

    dataloader_params = {
        "window" : 8*125*10,
        "step" : 1000,
        "frequency": 50,
        "batch_size" : 128
    }

    data_path = "./data/smartphone-decimeter-2022/"

    tasks = get_tasks_for_dataset()
    tasks = tasks[:2]

    print('Preprocessing tasks...')
    with Pool(6) as p:
        # generating combined_data
        res = [p.apply_async(generate_combined_data,args=(t,)) for t in tasks]
        for r in tqdm(res):
            r.get()

    with Pool(6) as p:
        # rotating data
        res = [p.apply_async(rotate_data,args=(t,)) for t in tasks]
        for idx,r in tqdm(enumerate(res),total=len(tasks)):
            r.get()

    model = LIT_TH_FGO_SimpleVDRModel(dataloader_params)

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



