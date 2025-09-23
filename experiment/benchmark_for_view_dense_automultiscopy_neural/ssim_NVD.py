import numpy as np
from cupyx.scipy.signal import convolve2d as cu_convolve2d
from math import exp
from PIL import Image
import cupy as cp

############################################################################### GPU


def gaussian(window_size, sigma):
    """Generate one-dimensional Gaussian window"""
    gauss = cp.array([exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
                     for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """Create two-dimensional Gaussian window"""
    _1D_window = gaussian(window_size, 1.5).reshape(-1, 1)
    _2D_window = cp.outer(_1D_window, _1D_window.T)
    return _2D_window.astype(cp.float32)  # Return 2D window, without adding extra dimensions

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Calculate SSIM core function"""
    # Normalize pixel values to 0~1 range
    img1 = img1.astype(cp.float32) / 255.0
    img2 = img2.astype(cp.float32) / 255.0
    
    # Fix: ensure image dimensions match
    # min_height = min(img1.shape[0], img2.shape[0])
    # min_width = min(img1.shape[1], img2.shape[1])
    # img1 = img1[:min_height, :min_width]
    # img2 = img2[:min_height, :min_width]
    
    # Add batch and channel dimensions (if needed)
    if len(img1.shape) == 2:  # Grayscale image
        img1 = img1[cp.newaxis, cp.newaxis, :, :]
        img2 = img2[cp.newaxis, cp.newaxis, :, :]
    elif len(img1.shape) == 3:  # Color image
        # Convert format to (C, H, W)
        img1 = img1.transpose(2, 0, 1)[cp.newaxis, :, :, :]
        img2 = img2.transpose(2, 0, 1)[cp.newaxis, :, :, :]
    
    # Update channel count
    _, channel, height, width = img1.shape


    def conv2d(input_img, kernel):
        
        result = cp.zeros_like(input_img)
        for b in range(input_img.shape[0]):  # Iterate through batches
            for c in range(input_img.shape[1]):  # Iterate through channels
                # Directly use 'same' mode convolution, no manual padding
                result[b, c] = cu_convolve2d(input_img[b, c], kernel, 
                                         mode='same', boundary='symm')
        return result

    # Calculate local means
    mu1 = conv2d(img1, window)
    mu2 = conv2d(img2, window)

    # Calculate intermediate terms
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Calculate local variance and covariance
    sigma1_sq = conv2d(img1*img1, window) - mu1_sq
    sigma2_sq = conv2d(img2*img2, window) - mu2_sq
    sigma12 = conv2d(img1*img2, window) - mu1_mu2

    # SSIM constants (R=1 after normalization)
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2

    # Calculate SSIM mapping
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)  # Avoid division by zero
    ssim_map = cp.clip(ssim_map, 0, 1)  # Limit to range 0~1

    # Return average value
    return cp.mean(ssim_map)

class SSIM:
    """SSIM calculation class"""
    def __init__(self, window_size=11, size_average=True):
        self.window_size = window_size
        self.size_average = size_average
        self.window = create_window(window_size, 1)  # Create 2D window
    
    def __call__(self, img1, img2):
        # Ensure image dimensions match
        if img1.shape != img2.shape:
            # Crop to the same size
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        return _ssim(img1, img2, self.window, 
                    self.window_size, 0, self.size_average)  # channel parameter has been removed

