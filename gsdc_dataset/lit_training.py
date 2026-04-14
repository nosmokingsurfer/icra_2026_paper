import torch
import pytorch_lightning as L
from  pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, BatchSizeFinder

import torch.functional as F
from  torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0,'.')
from gsdc_dataset.simple_vdr_model import SimpleVDRmodel
from gsdc_dataset.gsdc_dataloader import GSDC_dataset, generate_combined_data, rotate_data

import torch.utils.tensorboard

class LIT_SimpleVDRModel(L.LightningModule):
    def __init__(self):
        super(LIT_SimpleVDRModel,self).__init__()
        self.model = SimpleVDRmodel()
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

        imu = torch.concat((acc,gyro),dim=-1).swapaxes(-1,-2)

        pred = self.model(imu)

        window = torch.ones((2,1,8))/8
        window = window.to(gt_vel.device)
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)

        loss = torch.nn.functional.mse_loss(pred,gt_vel)
        self.log('train_loss', loss, on_epoch=True)
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
        self.log('val_loss', loss, on_epoch=True)
        return loss

class LIT_GSDC_datamodule(L.LightningDataModule):
    def __init__(self, tasks, data_path, dataloader_params):
        super().__init__()
        self.tasks = tasks
        self.data_path = data_path
        self.dataloader_params = dataloader_params
        self.hparams.update(dataloader_params)
        self.save_hyperparameters()

    def setup(self, stage):
        super().setup(stage)
        # TODO make split for train and val here

    def train_dataloader(self):
        dataset = GSDC_dataset('train', self.tasks, self.data_path, **self.dataloader_params)
        return DataLoader(dataset,self.dataloader_params['batch_size'], shuffle=True, drop_last=True)

    def val_dataloader(self):
        dataset = GSDC_dataset('val', self.tasks, self.data_path, **self.dataloader_params)
        return DataLoader(dataset,self.dataloader_params['batch_size'], shuffle=False, drop_last=True)

    def test_dataloader(self):
        return super().test_dataloader()


if __name__ == "__main__":
    model = LIT_SimpleVDRModel()

    # batch_size_finder = BatchSizeFinder() # TODO

    dataloader_params = {
        "window" : 8*125*10,
        "step" : 1000,
        "frequency": 50,
        "batch_size" : 128
    }

    data_path = "./data/smartphone-decimeter-2022/"
    imu_files = list(Path(data_path + "train/").rglob("**/device_imu.csv"))


    print("Indexing tasks...")
    tasks = []
    for t in tqdm(imu_files):
        sample_id = t.parts[-3].replace('-','_') + "_" + t.parts[-2]
        imu_file = str(t.parent / "device_imu.csv")
        gt_file = str(t.parent / "ground_truth.csv")

        tasks.append(
            {
                "sample_id" : sample_id,
                "mode" : "train",
                "imu_file" : imu_file,
                "gt_file" : gt_file
            }
        )


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



