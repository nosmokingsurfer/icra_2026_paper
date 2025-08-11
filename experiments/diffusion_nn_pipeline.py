# noisy_imu -> diffusion -> denoised_imu -> ronin -> velocity -> FGO -> trajectory -> loss 
# fgo nn splines input size before reshape torch.Size([64, 6, 3, 100]) and after reshape torch.Size([384, 3, 100])
#                                                       B, S, C, W                                  -1, C, W
# diffusion splines input and output size torch.Size([64, 3, 100])
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.diffusion_splines import IMUDenoiser
from spline_dataset.spline_dataloader import SplineIMUDataset
from ronin_resnet import ResNet1D, BasicBlock1D, FCOutputModule

class DiffussionNN(nn.Module):
    def __init__(self):
        super().__init__() 

        diffusion_config = {
            'spline_path': './out/splines',  # or path to your spline files
            'num_samples': 5000,  # if using synthetic data
            'window_size': 100,
            'noise_level': 0.2,
            'stride' : 20,
            'imu_freq': 100.0,
            'batch_size': 64,
            'num_workers': 0,
            'num_epochs': 100,
            'val_interval': 5,
            'T': 1000,  # diffusion timesteps
            'lr': 1e-4,
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model_state = torch.load(open('model.pt','rb'), device)
        self.diffusion_model = IMUDenoiser(
            input_channels=3,  # ax, ay, ωz
            window_size=diffusion_config['window_size']
        ).to(device)

        self.diffusion_model.load_state_dict(trained_model_state)

        self.nn_model = ResNet1D(
            num_inputs=3,          # Must match diffusion's output channels
            num_outputs=10,        # Your desired output classes/dimension
            block_type=BasicBlock1D,   # Or Bottleneck1D
            group_sizes=[2, 2, 2], # Example architecture (adjust as needed)
            base_plane=64          # Standard ResNet starting plane size
        )
        self.nn_model.load_state_dict(torch.load("out/model_fgo.pth"))

    def forward(self, x):
        x = self.diffusion_model(x)
        x = self.nn_model(x)
        return x
    

def training_pipeline():
    dataset = SplineIMUDataset(
        spline_path='./out/splines',
        num_samples=5000,
        window_size=100,
        mode='both',
        noise_level=0.2,
        imu_freq=100.0
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=6
    )