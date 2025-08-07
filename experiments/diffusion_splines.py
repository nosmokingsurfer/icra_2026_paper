import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from spline_dataset.spline_dataloader import SplineIMUDataset

# Define the UNet model for IMU denoising
class IMUDenoiser(nn.Module):
    def __init__(self, input_channels=3, window_size=20):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, input_channels, kernel_size=3, padding=1),
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
        )
        
        # Adaptive pooling to mix time information
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, t):
        # x shape: [batch, channels, window_size]
        # t shape: [batch]
        
        # Time embedding
        t = t.float().unsqueeze(-1)  # [batch, 1]
        t_emb = self.time_embed(t)  # [batch, 256]
        t_emb = t_emb.unsqueeze(-1)  # [batch, 256, 1]
        
        # Encode input
        x = self.encoder(x)
        
        # Add time information
        pooled = self.pool(x)  # [batch, 256, 1]
        x = x + t_emb * pooled
        
        # Middle processing
        x = self.middle(x)
        
        # Decode
        x = self.decoder(x)
        
        return x

# Diffusion process utilities
class DiffusionProcess:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def forward_diffusion(self, x0, t):
        """Apply noise to clean data at timestep t"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t]).view(-1, 1, 1)
        
        noise = torch.randn_like(x0)
        noisy = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        
        return noisy, noise
    
    def sample_timesteps(self, n):
        """Sample random timesteps for training"""
        return torch.randint(1, self.T, (n,))

# Training function
def train_diffusion_model(config):
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and dataloader
    dataset = SplineIMUDataset(
        spline_path=config['spline_path'],
        num_samples=config['num_samples'],
        window_size=config['window_size'],
        mode='both',
        noise_level=config['noise_level'],
        imu_freq=config['imu_freq']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=config['num_workers']
    )
    
    # Initialize model and diffusion process
    model = IMUDenoiser(
        input_channels=3,  # ax, ay, ωz
        window_size=config['window_size']
    ).to(device)
    
    diffusion = DiffusionProcess(T=config['T'])
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    # Tensorboard logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/diffusion_imu_{timestamp}")
    
    # Training loop
    global_step = 0
    for epoch in range(config['num_epochs']):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in progress_bar:
            clean = batch['clean'].to(device)
            noisy = batch['noisy'].to(device)
            batch_size = clean.shape[0]
            
            # Sample random timesteps
            t = diffusion.sample_timesteps(batch_size).to(device)
            
            # Forward diffusion (add noise to clean data)
            x_t, noise = diffusion.forward_diffusion(clean, t)
            
            # Predict noise
            predicted_noise = model(x_t, t)
            
            # Compute loss
            loss = criterion(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            global_step += 1
            writer.add_scalar("train/loss", loss.item(), global_step)
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Validation
        if (epoch + 1) % config['val_interval'] == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for val_batch in dataloader:
                    clean = val_batch['clean'].to(device)
                    noisy = val_batch['noisy'].to(device)
                    
                    # Use same timestep for validation (mid-range)
                    t = torch.full((clean.shape[0],), diffusion.T//2).to(device)
                    
                    x_t, noise = diffusion.forward_diffusion(clean, t)
                    predicted_noise = model(x_t, t)
                    
                    val_loss = criterion(predicted_noise, noise)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            writer.add_scalar("val/loss", avg_val_loss, global_step)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save model checkpoint
            if avg_val_loss == min(val_losses):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, f"best_model_{timestamp}.pt")
    
    writer.close()
    return model

# Sampling function (for generating denoised IMU)
@torch.no_grad()
def sample_from_diffusion(model, diffusion, noisy_input, device):
    """Denoise IMU data using the trained diffusion model"""
    model.eval()
    
    # Start from pure noise
    x = noisy_input.to(device)
    
    # Reverse diffusion process
    for t in range(diffusion.T, 0, -1):
        t_batch = torch.full((x.shape[0],), t).to(device)
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Remove predicted noise
        alpha_t = diffusion.alphas[t-1].to(device)
        alpha_bar_t = diffusion.alpha_bars[t-1].to(device)
        beta_t = diffusion.betas[t-1].to(device)
        
        if t > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * noise
    
    return x



# Example usage of sampling
def demonstrate_denoising(trained_model, diffusion, dataset, device):
    # Get a sample from the dataset
    sample = dataset[0]
    noisy = sample['noisy'].unsqueeze(0).to(device)
    clean = sample['clean'].unsqueeze(0).to(device)
    
    # Denoise using the trained model
    denoised = sample_from_diffusion(trained_model, diffusion, noisy, device)
    
    # Calculate metrics
    mse_noisy = torch.mean((noisy - clean)**2).item()
    mse_denoised = torch.mean((denoised - clean)**2).item()
    
    print(f"Original Noisy MSE: {mse_noisy:.4f}")
    print(f"After Denoising MSE: {mse_denoised:.4f}")
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"T = {diffusion.T}\n" + \
                 
    f"Original Noisy MSE: {mse_noisy:.4f}\n" + \
    f"After Denoising MSE: {mse_denoised:.4f}")

    titles = ['Acceleration X', 'Acceleration Y', 'Angular Velocity Z']
    
    for i in range(3):
        axes[i].plot(clean[0, i].cpu(), 'g-', label='Clean')
        axes[i].plot(noisy[0, i].cpu(), 'r:', label='Noisy')
        axes[i].plot(denoised[0, i].cpu(), 'b--', label='Denoised')
        axes[i].set_title(titles[i])
        axes[i].legend()
    
    plt.tight_layout()

    #TODO show the integrated velocity and trajectory - clean, noisy and denoised
    plt.show()




if __name__ == "__main__":
    # Configuration
    config = {
        'spline_path': './out/splines',  # or path to your spline files
        'num_samples': 5000,  # if using synthetic data
        'window_size': 200,
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

    if not  os.path.exists('model.pt'):
        # Train the model
        trained_model = train_diffusion_model(config)
        torch.save(trained_model.state_dict(),'model.pt')

    else:
        trained_model_state = torch.load(open('model.pt','rb'),'cpu')
        trained_model = IMUDenoiser(
            input_channels=3,  # ax, ay, ωz
            window_size=config['window_size']
        ).to('cpu')

        trained_model.load_state_dict(trained_model_state)




    dataset = SplineIMUDataset(
        spline_path=config['spline_path'],
        num_samples=config['num_samples'],
        window_size=config['window_size'],
        mode='diffusion',
        noise_level=config['noise_level'],
        imu_freq=config['imu_freq']
    )
    device='cpu'

    for T in range(1,30):
    # Demonstrate on a sample
        demonstrate_denoising(trained_model, DiffusionProcess(T), dataset, device)