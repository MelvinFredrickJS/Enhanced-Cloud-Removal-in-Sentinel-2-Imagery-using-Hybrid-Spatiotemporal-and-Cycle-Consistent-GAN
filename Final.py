import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
import itertools

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Directories
CLOUDY_DIR = r'C:\\Users\\Joseph\Desktop\\cloud removal\\FULL DATASET-20240913T153435Z-001\\FULL DATASET\\cloud'
CLOUD_FREE_DIR = r'C:\\Users\\Joseph\Desktop\\cloud removal\\FULL DATASET-20240913T153435Z-001\\FULL DATASET\\cloudfree'
OUTPUT_DIR = r'C:\\Users\\Joseph\Desktop\\cloud removal\\res'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3
BATCH_SIZE = 1
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 5.0
EPOCHS = 200
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Custom Dataset
class SatelliteDataset(Dataset):
    def _init_(self, cloudy_dir, cloud_free_dir):
        self.cloudy_paths = sorted(glob(os.path.join(cloudy_dir, '*.png')))
        self.cloud_free_paths = sorted(glob(os.path.join(cloud_free_dir, '*.png')))
        self.transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def _len_(self):
        return len(self.cloudy_paths)
    
    def _getitem_(self, idx):
        cloudy_img = Image.open(self.cloudy_paths[idx]).convert('RGB')
        cloud_free_img = Image.open(self.cloud_free_paths[idx]).convert('RGB')
        
        cloudy_img = self.transform(cloudy_img)
        cloud_free_img = self.transform(cloud_free_img)
        
        return {'cloudy': cloudy_img, 'cloud_free': cloud_free_img}

# ResNet block
class ResidualBlock(nn.Module):
    def _init_(self, features):
        super(ResidualBlock, self)._init_()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features)
        )
    
    def forward(self, x):
        return x + self.block(x)

# Generator
class Generator(nn.Module):
    def _init_(self):
        super(Generator, self)._init_()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(CHANNELS, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down_sampling = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(9)]
        )
        
        # Upsampling
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, CHANNELS, 7),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down_sampling(x)
        x = self.res_blocks(x)
        x = self.up_sampling(x)
        return self.output(x)

# Discriminator
class Discriminator(nn.Module):
    def _init_(self):
        super(Discriminator, self)._init_()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(CHANNELS, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_cloudy2clear = Generator().to(device)
G_clear2cloudy = Generator().to(device)
D_cloudy = Discriminator().to(device)
D_clear = Discriminator().to(device)

# Loss functions
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(
    itertools.chain(G_cloudy2clear.parameters(), G_clear2cloudy.parameters()),
    lr=LR, betas=(BETA1, BETA2)
)
optimizer_D_cloudy = optim.Adam(D_cloudy.parameters(), lr=LR, betas=(BETA1, BETA2))
optimizer_D_clear = optim.Adam(D_clear.parameters(), lr=LR, betas=(BETA1, BETA2))

# Learning rate schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LinearLR(optimizer_G, 
                                                  start_factor=1.0,
                                                  end_factor=0.0,
                                                  total_iters=EPOCHS)
lr_scheduler_D_cloudy = torch.optim.lr_scheduler.LinearLR(optimizer_D_cloudy,
                                                         start_factor=1.0,
                                                         end_factor=0.0,
                                                         total_iters=EPOCHS)
lr_scheduler_D_clear = torch.optim.lr_scheduler.LinearLR(optimizer_D_clear,
                                                        start_factor=1.0,
                                                        end_factor=0.0,
                                                        total_iters=EPOCHS)

# Create dataset and dataloader
dataset = SatelliteDataset(CLOUDY_DIR, CLOUD_FREE_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Training function
def train():
    for epoch in range(EPOCHS):
        for i, batch in enumerate(dataloader):
            real_cloudy = batch['cloudy'].to(device)
            real_clear = batch['cloud_free'].to(device)
            
            # Generate fake images
            fake_clear = G_cloudy2clear(real_cloudy)
            fake_cloudy = G_clear2cloudy(real_clear)
            
            # Train Generators
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_clear = criterion_identity(G_cloudy2clear(real_clear), real_clear)
            loss_id_cloudy = criterion_identity(G_clear2cloudy(real_cloudy), real_cloudy)
            loss_identity = (loss_id_clear + loss_id_cloudy) * LAMBDA_IDENTITY
            
            # GAN loss
            pred_fake_clear = D_clear(fake_clear)
            loss_GAN_cloudy2clear = criterion_GAN(pred_fake_clear, torch.ones_like(pred_fake_clear))
            
            pred_fake_cloudy = D_cloudy(fake_cloudy)
            loss_GAN_clear2cloudy = criterion_GAN(pred_fake_cloudy, torch.ones_like(pred_fake_cloudy))
            
            # Cycle loss
            recovered_cloudy = G_clear2cloudy(fake_clear)
            loss_cycle_cloudy = criterion_cycle(recovered_cloudy, real_cloudy)
            
            recovered_clear = G_cloudy2clear(fake_cloudy)
            loss_cycle_clear = criterion_cycle(recovered_clear, real_clear)
            
            loss_cycle = (loss_cycle_cloudy + loss_cycle_clear) * LAMBDA_CYCLE
            
            # Total generator loss
            loss_G = loss_GAN_cloudy2clear + loss_GAN_clear2cloudy + loss_cycle + loss_identity
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator Clear
            optimizer_D_clear.zero_grad()
            pred_real_clear = D_clear(real_clear)
            loss_D_real = criterion_GAN(pred_real_clear, torch.ones_like(pred_real_clear))
            pred_fake_clear = D_clear(fake_clear.detach())
            loss_D_fake = criterion_GAN(pred_fake_clear, torch.zeros_like(pred_fake_clear))
            loss_D_clear = (loss_D_real + loss_D_fake) * 0.5
            loss_D_clear.backward()
            optimizer_D_clear.step()
            
            # Train Discriminator Cloudy
            optimizer_D_cloudy.zero_grad()
            pred_real_cloudy = D_cloudy(real_cloudy)
            loss_D_real = criterion_GAN(pred_real_cloudy, torch.ones_like(pred_real_cloudy))
            pred_fake_cloudy = D_cloudy(fake_cloudy.detach())
            loss_D_fake = criterion_GAN(pred_fake_cloudy, torch.zeros_like(pred_fake_cloudy))
            loss_D_cloudy = (loss_D_real + loss_D_fake) * 0.5
            loss_D_cloudy.backward()
            optimizer_D_cloudy.step()
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {(loss_D_clear + loss_D_cloudy).item():.4f}] "
                      f"[G loss: {loss_G.item():.4f}]")
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_cloudy.step()
        lr_scheduler_D_clear.step()
        
        # Save models periodically
        if (epoch + 1) % 50 == 0:
            torch.save({
                'G_cloudy2clear': G_cloudy2clear.state_dict(),
                'G_clear2cloudy': G_clear2cloudy.state_dict(),
                'D_cloudy': D_cloudy.state_dict(),
                'D_clear': D_clear.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_cloudy': optimizer_D_cloudy.state_dict(),
                'optimizer_D_clear': optimizer_D_clear.state_dict()
            }, f'checkpoint_epoch_{epoch+1}.pth')

# Function to save generated images
def save_images(epoch, batch_idx, real_cloudy, real_clear, fake_clear):
    images = torch.cat([real_cloudy, real_clear, fake_clear], dim=3)
    save_path = os.path.join(OUTPUT_DIR, f'epoch_{epoch}batch{batch_idx}.png')
    torchvision.utils.save_image(images, save_path, normalize=True)

if __name__ == '_main_':
    print("Starting training...")
    train()
    print("Training finished!")