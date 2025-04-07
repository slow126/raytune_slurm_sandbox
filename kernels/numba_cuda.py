import numpy as np
import numba
from numba import cuda
import math
import torch
# CUDA kernel function
@cuda.jit(
        "void(float32[:,:,:,:], float32[:,:,:,:])"
)
def add_checkerboard_kernel(images, output):
    """
    Add an RGB checkerboard pattern to images: red, green, blue pattern repeating.
    
    Args:
        images: Input array with shape (batch_size, height, width, channels)
        output: Output array with same shape as images
    """
    # Get thread indices
    batch_idx = cuda.blockIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.z * cuda.blockDim.x + cuda.threadIdx.x
    
    # Get array dimensions
    batch_size, height, width, channels = images.shape
    
    # Check if indices are within bounds
    if batch_idx < batch_size and row < height and col < width:
        # Determine color based on position (red=0, green=1, blue=2)
        color_idx = (row + col) % 3
        
        # Process all channels for this pixel
        for c in range(channels):
            # Add 1.0 to the channel that matches our color pattern, leave others unchanged
            if c == color_idx:
                output[batch_idx, row, col, c] = images[batch_idx, row, col, c] + 1.0
            else:
                output[batch_idx, row, col, c] = images[batch_idx, row, col, c]

# Helper function with device decorator
@cuda.jit(device=True)
def is_even(num):
    """Helper function to check if a number is even."""
    return num % 2 == 0

# Function to process images on GPU
def process_images_cuda(images):
    """
    Process a batch of images by adding an RGB checkerboard pattern.
    
    Args:
        images: NumPy array with shape (batch_size, height, width, channels)
        
    Returns:
        Processed images with RGB checkerboard pattern added
    """
    # Make sure input is in the right format (batch, height, width, channels)
    if images.ndim != 4:
        raise ValueError("Input must be 4D array: (batch_size, height, width, channels)")
    
    # Get dimensions
    batch_size, height, width, channels = images.shape
    
    # Check if input is already on GPU (PyTorch tensor on CUDA)
    if isinstance(images, torch.Tensor) and images.is_cuda:
        # If already on GPU, use torch functions
        output = torch.zeros_like(images)
        d_images = images  # Already on device
        d_output = output  # Already on device
    else:
        # For NumPy arrays or CPU tensors, use numba's CUDA functions
        output = np.zeros_like(images)
        d_images = cuda.to_device(images)
        d_output = cuda.to_device(output)
    
    # Set up grid and block dimensions
    threads_per_block = (32, 32, 1)  # Add z-dimension to thread block
    blocks_per_grid_y = math.ceil(height / threads_per_block[0])
    blocks_per_grid_z = math.ceil(width / threads_per_block[1])
    blocks_per_grid = (batch_size, blocks_per_grid_y, blocks_per_grid_z)
    
    # Launch kernel
    add_checkerboard_kernel[blocks_per_grid, threads_per_block](d_images, d_output)
    
    return d_output


if __name__ == "__main__":
    # Generate random input data
    batch_size = 4
    height = 1024
    width = 1024
    channels = 3

    # Create random input data
    # Increase batch size for testing multiple images

    # Create a zero-filled array for easy debugging
    images = np.zeros((batch_size, height, width, channels), dtype=np.float32)
    print(f"Created test array with shape: {images.shape}")

    # Process images on GPU
    processed_images = process_images_cuda(images)
    print(f"Processed images with shape: {processed_images.shape}")

    # Print the first image in the batch
    print(f"First image in the batch: {processed_images[0, :10, :10, :]}")

    # Display the first image from the batch using torchvision
    import torch
    import torchvision
    
    # Convert numpy array to torch tensor
    tensor_images = torch.from_numpy(processed_images.copy_to_host())
    tensor_image = tensor_images[0]
    
    # Ensure the tensor is in the right format for torchvision (C, H, W)
    if tensor_image.shape[-1] == 3:  # If the channels are in the last dimension (H, W, C)
        tensor_image = tensor_image.permute(2, 0, 1)  # Convert to (C, H, W)
    
    # Normalize if needed (assuming values are in [0, 1] range)
    tensor_image = tensor_image.clamp(0, 1)
    
    # Display the image
    torchvision.utils.save_image(tensor_image, 'first_image.png')
    print("First image saved as 'first_image.png'")


    # Process images on GPU
