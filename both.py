import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from models.ST_gan import Generator  # Import ST-GAN Generator
from models.cycle_gan import CloudRemovalModel  # Import Cycle-GAN model
class CloudDataset(Dataset):
    def __init__(self, cloudy_dir, clear_dir, transform=None):
        self.cloudy_dir = cloudy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        
        self.cloudy_images = sorted([f for f in os.listdir(cloudy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.clear_images = sorted([f for f in os.listdir(clear_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.cloudy_images)
    
    def __getitem__(self, idx):
        cloudy_path = os.path.join(self.cloudy_dir, self.cloudy_images[idx])
        clear_path = os.path.join(self.clear_dir, self.clear_images[idx])
        
        cloudy_img = Image.open(cloudy_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        
        if self.transform:
            cloudy_img = self.transform(cloudy_img)
            clear_img = self.transform(clear_img)
            
        return cloudy_img, clear_img
class CombinedCloudRemoval:
    def __init__(self, device):
        self.device = device
        
        # Initialize ST-GAN components
        self.st_generator = Generator(input_channels=6).to(device)  # From first model
        
        # Initialize Cycle-GAN components
        self.cycle_model = CloudRemovalModel(device)  # From second model
        
        # Initialize optimizers
        self.st_optimizer = torch.optim.Adam(
            self.st_generator.parameters(),
            lr=0.0001,
            betas=(0.5, 0.999)
        )
        
        # Loss functions
        self.criterion_pixel = nn.L1Loss()
    
    def train_step(self, cloudy_imgs, clear_imgs):
        # Step 1: ST-GAN Forward Pass
        # Combine cloudy and clear images for ST-GAN input
        combined_input = torch.cat((cloudy_imgs, clear_imgs), dim=1)
        st_gan_output = self.st_generator(combined_input)
        
        # Step 2: Cycle-GAN Forward Pass
        # Use ST-GAN output and original cloudy image as inputs for Cycle-GAN
        cycle_gan_losses = self.cycle_model.train_step(cloudy_imgs, st_gan_output)
        
        # Calculate ST-GAN losses
        st_loss_pixel = self.criterion_pixel(st_gan_output, clear_imgs)
        st_loss = 100 * st_loss_pixel
        
        # Update ST-GAN
        self.st_optimizer.zero_grad()
        st_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.st_generator.parameters(), max_norm=1.0)
        self.st_optimizer.step()
        
        # Combine losses for monitoring
        total_losses = {
            'st_loss': st_loss.item(),
            'cycle_gan_g_loss': cycle_gan_losses['loss_G'],
            'cycle_gan_d_loss': cycle_gan_losses['loss_D'],
            'cycle_loss': cycle_gan_losses['loss_cycle'],
            'identity_loss': cycle_gan_losses['loss_identity']
        }
        
        return total_losses, st_gan_output
    
    def save_checkpoint(self, epoch, output_dir):
        # Save both models' checkpoints
        st_checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.st_generator.state_dict(),
            'optimizer_state_dict': self.st_optimizer.state_dict(),
        }
        torch.save(st_checkpoint, os.path.join(output_dir, f'st_gan_checkpoint_epoch_{epoch}.pth'))
        
        # Save Cycle-GAN checkpoint using its own method
        self.cycle_model.save_checkpoint(epoch, output_dir)
    
    def load_checkpoint(self, st_checkpoint_path, cycle_checkpoint_path):
        # Load ST-GAN checkpoint
        st_checkpoint = torch.load(st_checkpoint_path)
        self.st_generator.load_state_dict(st_checkpoint['generator_state_dict'])
        self.st_optimizer.load_state_dict(st_checkpoint['optimizer_state_dict'])
        
        # Load Cycle-GAN checkpoint using its own method
        cycle_epoch = self.cycle_model.load_checkpoint(cycle_checkpoint_path)
        
        return st_checkpoint['epoch'], cycle_epoch

def train_combined_model(cloudy_dir, clear_dir, output_dir, num_epochs=100, batch_size=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset and dataloader
    dataset = CloudDataset(cloudy_dir, clear_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize combined model
    model = CombinedCloudRemoval(device)
    
    try:
        for epoch in range(num_epochs):
            for i, (cloudy_imgs, clear_imgs) in enumerate(dataloader):
                # Clear cache periodically
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Move data to device
                cloudy_imgs = cloudy_imgs.to(device)
                clear_imgs = clear_imgs.to(device)
                
                # Train model
                losses, st_output = model.train_step(cloudy_imgs, clear_imgs)
                
                # Print progress
                if i % 10 == 0:
                    print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[ST Loss: {losses['st_loss']:.4f}] "
                          f"[Cycle-G Loss: {losses['cycle_gan_g_loss']:.4f}] "
                          f"[Cycle-D Loss: {losses['cycle_gan_d_loss']:.4f}]")
                    
                    # Save sample images
                    if i % 100 == 0:
                        for j in range(min(batch_size, 3)):  # Save up to 3 images
                            # Save ST-GAN intermediate output
                            save_generated_image(
                                st_output[j],
                                os.path.join(output_dir, 'results', f'st_gan_epoch_{epoch}_batch_{i}_img_{j}.png')
                            )
                            
                            # Final output will be saved by Cycle-GAN's own saving mechanism
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.save_checkpoint(
                    epoch,
                    os.path.join(output_dir, 'checkpoints')
                )
                
    except RuntimeError as e:
        print(f"Runtime error occurred: {e}")
        print("Trying to free up memory...")
        torch.cuda.empty_cache()
        
    return model

if __name__ == "__main__":
    # Set paths
    cloudy_dir = r"C:\\Users\\Joseph\Desktop\\cloud removal\ST-GAN\\input\\cloud"
    clear_dir =  r"C:\\Users\\Joseph\Desktop\\cloud removal\ST-GAN\\input\\cloudfree"
    output_dir = r"C:\\Users\\Joseph\Desktop\\cloud removal\ST-GAN\\hybrid out"
    
    # Train the combined model
    model = train_combined_model(cloudy_dir, clear_dir, output_dir)