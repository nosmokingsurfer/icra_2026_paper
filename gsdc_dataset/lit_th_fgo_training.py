import torch
import pytorch_lightning as L
from  pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, BatchSizeFinder

import torch.functional as F
from  torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import theseus as th

from gsdc_dataset.th_simple_model import FGO_SimpleVDRModel
from gsdc_dataset.gsdc_dataloader import GSDC_dataset

import torch.utils.tensorboard

class LIT_TH_FGO_SimpleVDRModel(L.LightningModule):
    def __init__(self,**dataloader_params):
        super(LIT_TH_FGO_SimpleVDRModel,self).__init__()
        self.dataloader_params = dataloader_params
        self.model = FGO_SimpleVDRModel(**dataloader_params)


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
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)
        gt_traj = torch.conv1d(gt_traj.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)

        window=torch.ones((1,1,8))/8
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

        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X,y = val_batch
        pred = self.model(X)
        loss = F.loss.mse_loss(pred,y)
        self.log('val_loss', loss, on_epoch=True)


if __name__ == "__main__":
    dataloader_params = {
        "window" : 8*125*10,
        "step" : 200,
        "frequency": 50,
        "batch_size" : 16

    }


    model = LIT_TH_FGO_SimpleVDRModel(**dataloader_params)

    # batch_size_finder = BatchSizeFinder() # TODO




    gsdc_train_dataset = GSDC_dataset('train', **dataloader_params)
    # gsdc_val_dataset = GSDC_dataset('test', **dataloader_params)


    lr_monitor = LearningRateMonitor('epoch')

    checkpoint_monitor = ModelCheckpoint(
        filename='{epoch}-{val_loss:.5f}-{train_loss:.5f}',
        save_top_k=5,
        save_last=True,
        monitor='train_loss'
        )


    trainer = L.Trainer(
        accelerator='auto',
        max_epochs=100,
        callbacks = [
            lr_monitor,
            checkpoint_monitor,
            ],
    )

    train_loader = DataLoader(gsdc_train_dataset,batch_size=dataloader_params['batch_size'],shuffle=True,drop_last=True)

    trainer.fit(model, train_loader)



