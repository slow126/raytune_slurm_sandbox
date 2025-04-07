import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from einops import rearrange
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")
if ROOT_DIR is not None:
    sys.path.append(ROOT_DIR)

from kernels.numba_cuda import process_images_cuda
from torchvision.utils import make_grid
from super_image import EdsrModel, ImageLoader

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)  # (B, emb_dim, H//patch_size, W//patch_size)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (B, num_patches, emb_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(
            in_features=emb_dim,
            hidden_features=int(emb_dim * mlp_ratio),
            out_features=emb_dim,
            dropout=dropout
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(
        self, 
        img_size=256,
        in_channels=3, 
        patch_size=16, 
        emb_dim=512, 
        num_layers=8,
        num_heads=8, 
        mlp_ratio=4.0, 
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))

        
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(emb_dim)
        
        # Decoder for image reconstruction
        self.decoder_embed = nn.Linear(emb_dim, patch_size * patch_size * in_channels)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Encoding
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Decoding (skip the cls token)
        x = x[:, 1:]  # Remove cls token
        
        # Project from embedding dimension to patch pixels
        x = self.decoder_embed(x)
        
        # Reshape to batch of flattened patches
        x = x.reshape(B, self.num_patches, self.patch_size, self.patch_size, self.in_channels)
        
        # Rearrange to image
        x = x.permute(0, 1, 4, 2, 3)  # B, num_patches, C, patch_size, patch_size
        
        # Reconstruct image by arranging patches
        reconstructed = torch.zeros(B, C, H, W, device=x.device)
        
        patches_per_row = H // self.patch_size
        for i in range(self.num_patches):
            row = i // patches_per_row
            col = i % patches_per_row
            
            h_start = row * self.patch_size
            w_start = col * self.patch_size
            
            reconstructed[:, :, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size] = x[:, i]
        
        return reconstructed

class ViTLightning(pl.LightningModule):
    def __init__(
        self,
        img_size=128,
        in_channels=3,
        patch_size=16,
        emb_dim=512,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=32,
        num_workers=4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViT(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.upsample_factor = int(img_size // 32) # 32 is the size of the CIFAR-10 images
        self.upsampler = EdsrModel.from_pretrained('eugenesiow/edsr', scale=self.upsample_factor)
        self.num_workers = num_workers
        self.data_dir = '/home/spencer/Deployments/raytune_slurm_sandbox/data'
    
    def forward(self, x):
        return self.model(x)
    
    def apply_cuda_kernel(self, x):
        # Convert to numpy, apply CUDA kernel, convert back to tensor
            
        # Convert to numpy and change to NHWC format for CUDA kernel
        x = x.permute(0, 2, 3, 1)
        
        # Apply CUDA kernel
        processed_tensor = process_images_cuda(x)
        
        # Convert back to tensor and NCHW format
        processed_tensor = processed_tensor.permute(0, 3, 1, 2)
        
        return processed_tensor.to(x.device)

    def convert_int_to_float(self, x):
        # Convert images to float and normalize to range [0, 1]
        if x.dtype != torch.float32:
            x = x.float()        
        # Normalize to [0, 1] if not already in that range
        if x.max() > 1.0 or x.min() < 0.0:
            x = x / 255.0 if x.max() > 1.0 else x
        return x
    
    def step(self, x):
        # Convert images to float and normalize to range [0, 1]
        with torch.no_grad():            
            # Upscale each image in the batch individually and clear memory after each
            upscaled_x = []
            # Initialize per-channel min and max values
            num_channels = x.size(1)
            channel_mins = torch.zeros(num_channels, device=x.device)
            channel_maxs = torch.zeros(num_channels, device=x.device)
            
            for i in range(x.size(0)):
                # Process one image at a time
                upscaled = self.upsampler(x[i:i+1].squeeze(0))
                upscaled_x.append(upscaled)
                
                # # Track min and max per channel
                # for c in range(num_channels):
                #     channel_min = upscaled[:, c].min()
                #     channel_max = upscaled[:, c].max()
                    
                #     # Update global min/max for each channel
                #     if i == 0:
                #         channel_mins[c] = channel_min
                #         channel_maxs[c] = channel_max
                #     else:
                #         channel_mins[c] = torch.min(channel_mins[c], channel_min)
                #         channel_maxs[c] = torch.max(channel_maxs[c], channel_max)
            
            # Stack all upscaled images
            x = torch.stack(upscaled_x, dim=0)
            del upscaled_x

            processed_x = x.clone()
            
            # Apply CUDA kernel to the input batch
            # x = self.apply_cuda_kernel(x)
            
            # Normalize each channel using the per-channel min and max values
            # for c in range(num_channels):
            #     x[:, c] = (x[:, c] - channel_mins[c]) / (channel_maxs[c] - channel_mins[c] + 1e-8)

        # Get reconstruction and original
        reconstructed = self(x)
        
        # Compute reconstruction loss (MSE)
        loss = F.mse_loss(reconstructed, processed_x)  

        return loss, processed_x, reconstructed

    
    def training_step(self, batch, batch_idx):
        x, _ = batch  # We don't need labels for reconstruction

        loss, processed_x, reconstructed = self.step(x)
        
        # Log metrics
        self.log('train/train_loss', loss.detach(), prog_bar=True)
        
        # Log images to TensorBoard periodically
        if batch_idx % 100 == 0:
            # Create a grid of original images
            grid_original = make_grid(x[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('train/original_images', grid_original, self.global_step)
            
            # Create a grid of processed images
            grid_processed = make_grid(processed_x[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('train/processed_images', grid_processed, self.global_step)
            
            # Create a grid of reconstructed images
            grid_reconstructed = make_grid(reconstructed[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('train/reconstructed_images', grid_reconstructed, self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch  # We don't need labels for reconstruction

        loss, original_x, reconstructed = self.step(x)
        
        # Log metrics
        self.log('val/val_loss', loss.detach(), prog_bar=True)
        
        # Log images to TensorBoard at the end of validation
        if batch_idx == 0:
            # Create a grid of original images
            grid_original = make_grid(original_x[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('val/original_images', grid_original, self.global_step)
            
            # Create a grid of processed images
            grid_processed = make_grid(x[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('val/processed_images', grid_processed, self.global_step)
            
            # Create a grid of reconstructed images
            grid_reconstructed = make_grid(reconstructed[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('val/reconstructed_images', grid_reconstructed, self.global_step)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _ = batch  # We don't need labels for reconstruction

        loss, original_x, reconstructed = self.step(x)
        
        # Log metrics
        self.log('test/test_loss', loss.detach(), prog_bar=True)
        
        # Log images to TensorBoard at the end of validation
        if batch_idx == 0:
            # Create a grid of original images
            grid_original = make_grid(original_x[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('test/original_images', grid_original, self.global_step)
            
            # Create a grid of processed images
            grid_processed = make_grid(x[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('test/processed_images', grid_processed, self.global_step)
            
            # Create a grid of reconstructed images
            grid_reconstructed = make_grid(reconstructed[:8], nrow=4, normalize=True)
            self.logger.experiment.add_image('test/reconstructed_images', grid_reconstructed, self.global_step)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def prepare_data(self):
        # Download CIFAR-10 dataset
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)
        
    def setup(self, resize_size=32, stage=None):
        # Define upscaling transforms for CIFAR-10 (32x32) to resize_size x resize_size
        # Using bicubic interpolation for better quality upscaling
        transform_train = transforms.Compose([
            transforms.RandomCrop(resize_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        
        # Load CIFAR-10 datasets
        if stage == 'fit' or stage is None:
            cifar_train = datasets.CIFAR10(root=self.data_dir, train=True, transform=transform_train)
            cifar_val = datasets.CIFAR10(root=self.data_dir, train=False, transform=transform_val)
            
            # Use the full training set
            self.train_dataset = cifar_train
            self.val_dataset = cifar_val
            
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, transform=transform_val)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    # Initialize model and trainer
    model = ViTLightning(batch_size=64, num_workers=8)
    
    # Initialize trainer with TensorBoard logger
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=pl.loggers.TensorBoardLogger('lightning_logs', name='vit_reconstruction')
    )
    
    # Train model
    training_results = trainer.fit(model)
    
    # Test model
    test_results = trainer.test(model)

    print(training_results)
    print(test_results)
