import torch
import pytorch_lightning as L
from  pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, BatchSizeFinder
import torch.functional as F
from  torch.utils.data import DataLoader

from gsdc_dataset.simple_vdr_model import SimpleVDRmodel
from gsdc_dataset.gsdc_dataloader import GSDC_dataset

class LIT_SimpleVDRModel(L.LightningModule):
    def __init__(self):
        super(LIT_SimpleVDRModel,self).__init__()
        self.model = SimpleVDRmodel()


    def forward(self,X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def training_step(self, train_batch, batch_idx):
        acc = train_batch['acc']
        gyro = train_batch['gyro']
        gt_vel = train_batch['gt_velocity']
        gt_traj = train_batch['gt_traj']

        imu = torch.concat((acc,gyro),dim=-1).swapaxes(-1,-2)

        pred = self.model(imu)

        window = torch.ones((2,1,8))/8
        
        gt_vel = torch.conv1d(gt_vel.swapaxes(-1,-2),window,stride=8,groups=2).swapaxes(-1,-2)

        loss = torch.nn.functional.mse_loss(pred,gt_vel)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        X,y = val_batch
        pred = self.model(X)
        loss = F.loss.mse_loss(pred,y)
        self.log('val_loss', loss, on_epoch=True)


if __name__ == "__main__":
    model = LIT_SimpleVDRModel()

    # batch_size_finder = BatchSizeFinder() # TODO

    dataloader_params = {
        "window" : 2000,
        "step" : 200,
        "frequency": 50
    }


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
        max_epochs=20,
        callbacks = [
            lr_monitor,
            checkpoint_monitor,
            ]
    )

    train_loader = DataLoader(gsdc_train_dataset,batch_size=64,shuffle=True)

    trainer.fit(model, train_loader)



