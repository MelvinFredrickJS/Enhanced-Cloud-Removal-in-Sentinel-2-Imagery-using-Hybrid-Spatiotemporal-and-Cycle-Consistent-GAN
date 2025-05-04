import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# ===========================================================================
# ST-GAN Model Components
# ===========================================================================
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.conv_q = nn.Conv2d(channels, channels // 8, 1)
        self.conv_k = nn.Conv2d(channels, channels // 8, 1)
        self.conv_v = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        q = self.conv_q(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.conv_k(x).view(b, -1, h*w)
        v = self.conv_v(x).view(b, -1, h*w)
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        return out.view(b, c, h, w)

class STGenerator(nn.Module):
    def __init__(self, input_channels=6):
        super(STGenerator, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.temporal_attention = TemporalAttention(64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        att = self.temporal_attention(e2)
        return self.dec1(att)

class STDiscriminator(nn.Module):
    def __init__(self):
        super(STDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        features = self.conv(x)
        return self.classifier(features).view(-1, 1)

# ===========================================================================
# Cycle-GAN Model Components
# ===========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class CycleGenerator(nn.Module):
    def __init__(self, input_channels=3):
        super(CycleGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.residual = nn.Sequential(
            *[ResidualBlock(512) for _ in range(4)]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x

class CycleDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(CycleDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

# ===========================================================================
# Cycle-GAN Model Class
# ===========================================================================
class CloudRemovalModel:
    def __init__(self, device):
        self.device = device
        self.G_cloudy_to_clear = CycleGenerator().to(device)
        self.G_clear_to_cloudy = CycleGenerator().to(device)
        self.D_cloudy = CycleDiscriminator().to(device)
        self.D_clear = CycleDiscriminator().to(device)
        self.g_optimizer = torch.optim.Adam(
            list(self.G_cloudy_to_clear.parameters()) + 
            list(self.G_clear_to_cloudy.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            list(self.D_cloudy.parameters()) + 
            list(self.D_clear.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.criterion_GAN = nn.BCELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.G_cloudy_to_clear.load_state_dict(checkpoint['G_cloudy_to_clear'])
        self.G_clear_to_cloudy.load_state_dict(checkpoint['G_clear_to_cloudy'])
        self.D_cloudy.load_state_dict(checkpoint['D_cloudy'])
        self.D_clear.load_state_dict(checkpoint['D_clear'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        return checkpoint['epoch']

# ===========================================================================
# Combined Model Class
# ===========================================================================
class CombinedCloudRemoval:
    def __init__(self, device):
        self.device = device
        self.st_generator = STGenerator(input_channels=6).to(device)
        self.st_discriminator = STDiscriminator().to(device)
        self.cycle_model = CloudRemovalModel(device)
        self.st_g_optimizer = torch.optim.Adam(
            self.st_generator.parameters(),
            lr=0.0001,
            betas=(0.5, 0.999)
        )
        self.st_d_optimizer = torch.optim.Adam(
            self.st_discriminator.parameters(),
            lr=0.0001,
            betas=(0.5, 0.999)
        )
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_pixel = nn.L1Loss()
    
    def load_checkpoint(self, st_checkpoint_path, cycle_checkpoint_path):
        st_checkpoint = torch.load(st_checkpoint_path, weights_only=False)
        self.st_generator.load_state_dict(st_checkpoint['generator_state_dict'])
        self.st_discriminator.load_state_dict(st_checkpoint['discriminator_state_dict'])
        self.st_g_optimizer.load_state_dict(st_checkpoint['g_optimizer_state_dict'])
        self.st_d_optimizer.load_state_dict(st_checkpoint['d_optimizer_state_dict'])
        cycle_epoch = self.cycle_model.load_checkpoint(cycle_checkpoint_path)
        return st_checkpoint['epoch'], cycle_epoch

# ===========================================================================
# Inference Function
# ===========================================================================
def generate_cloud_free_image(checkpoint_dir, cloudy_image_path, output_image_path, device='cuda'):
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = CombinedCloudRemoval(device)
    
    # Load the latest checkpoint
    cycle_checkpoint_path = os.path.join(checkpoint_dir, 'cycle_gan_checkpoint_epoch_99.pth')
    st_checkpoint_path = os.path.join(checkpoint_dir, 'st_gan_checkpoint_epoch_99.pth')
    model.load_checkpoint(st_checkpoint_path, cycle_checkpoint_path)
    
    # Set model to evaluation mode
    model.st_generator.eval()
    model.cycle_model.G_cloudy_to_clear.eval()
    
    # Define transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load and preprocess cloudy image
    cloudy_img = Image.open(cloudy_image_path).convert('RGB')
    cloudy_tensor = transform(cloudy_img).unsqueeze(0).to(device)
    
    # Generate cloud-free image
    with torch.no_grad():
        combined_input = torch.cat((cloudy_tensor, cloudy_tensor), dim=1)
        st_output = model.st_generator(combined_input)
        final_output = model.cycle_model.G_cloudy_to_clear(cloudy_tensor)
    
    # Convert output tensor to image
    final_output = final_output.squeeze(0).cpu()
    final_output = (final_output * 0.5 + 0.5) * 255
    final_output = final_output.byte().permute(1, 2, 0).numpy()
    Image.fromarray(final_output).save(output_image_path)
    
    print(f"Cloud-free image saved to {output_image_path}")

# Example usage
if __name__ == "__main__":
    checkpoint_dir = r'C:\\Users\\Joseph\\Desktop\\project\\cloud removal\\res\\checkpoints\\old-both'
    cloudy_image_path = r'C:\\Users\\Joseph\\Desktop\\project\\cloud removal\\cloud\\46.png'
    output_image_path = r'C:\\Users\\Joseph\\Desktop\\project\\cloud removal\\res\\cloud_free_output.jpg'
    
    generate_cloud_free_image(checkpoint_dir, cloudy_image_path, output_image_path)