import os
import sys
import torch
from torchvision import datasets
from super_image import EdsrModel

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")
if ROOT_DIR is not None:
    sys.path.append(ROOT_DIR)

def download_cifar10(data_dir='./data'):
    """Download CIFAR-10 dataset"""
    print(f"Downloading CIFAR-10 dataset to {data_dir}...")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download training and test datasets
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)
    
    print("CIFAR-10 dataset downloaded successfully.")

def download_upscaling_model():
    """Download the EDSR upscaling model"""
    print("Downloading EDSR upscaling model...")
    
    # This will download and cache the model
    upscaler = EdsrModel.from_pretrained('eugenesiow/edsr', scale=4)
    
    print("EDSR upscaling model downloaded successfully.")

if __name__ == "__main__":
    # Set the data directory
    data_dir = './data'
    
    # Download the datasets and models
    download_cifar10(data_dir)
    download_upscaling_model()
    
    print("All downloads completed successfully.")
