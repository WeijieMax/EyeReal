import numpy as np
from cupyx.scipy.signal import convolve2d as cu_convolve2d
from math import exp
from PIL import Image
import cupy as cp

############################################################################### GPU


def gaussian(window_size, sigma):
    """生成一维高斯窗口"""
    gauss = cp.array([exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
                     for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """创建二维高斯窗口"""
    _1D_window = gaussian(window_size, 1.5).reshape(-1, 1)
    _2D_window = cp.outer(_1D_window, _1D_window.T)
    return _2D_window.astype(cp.float32)  # 返回2D窗口，不添加额外维度

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """计算SSIM核心函数"""
    # 像素值归一化到0~1范围
    img1 = img1.astype(cp.float32) / 255.0
    img2 = img2.astype(cp.float32) / 255.0
    
    # 修复：确保图像尺寸匹配
    # min_height = min(img1.shape[0], img2.shape[0])
    # min_width = min(img1.shape[1], img2.shape[1])
    # img1 = img1[:min_height, :min_width]
    # img2 = img2[:min_height, :min_width]
    
    # 添加批次和通道维度（如果需要）
    if len(img1.shape) == 2:  # 灰度图
        img1 = img1[cp.newaxis, cp.newaxis, :, :]
        img2 = img2[cp.newaxis, cp.newaxis, :, :]
    elif len(img1.shape) == 3:  # 彩色图
        # 转换格式为 (C, H, W)
        img1 = img1.transpose(2, 0, 1)[cp.newaxis, :, :, :]
        img2 = img2.transpose(2, 0, 1)[cp.newaxis, :, :, :]
    
    # 更新通道数
    _, channel, height, width = img1.shape


    def conv2d(input_img, kernel):
        
        result = cp.zeros_like(input_img)
        for b in range(input_img.shape[0]):  # 遍历批次
            for c in range(input_img.shape[1]):  # 遍历通道
                # 直接使用'same'模式卷积，不手动填充
                result[b, c] = cu_convolve2d(input_img[b, c], kernel, 
                                         mode='same', boundary='symm')
        return result

    # 计算局部均值
    mu1 = conv2d(img1, window)
    mu2 = conv2d(img2, window)

    # 计算中间项
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算局部方差和协方差
    sigma1_sq = conv2d(img1*img1, window) - mu1_sq
    sigma2_sq = conv2d(img2*img2, window) - mu2_sq
    sigma12 = conv2d(img1*img2, window) - mu1_mu2

    # SSIM常数（归一化后R=1）
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2

    # 计算SSIM映射
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)  # 避免除零错误
    ssim_map = cp.clip(ssim_map, 0, 1)  # 限制在0~1之间

    # 返回平均值
    return cp.mean(ssim_map)

class SSIM:
    """SSIM计算类"""
    def __init__(self, window_size=11, size_average=True):
        self.window_size = window_size
        self.size_average = size_average
        self.window = create_window(window_size, 1)  # 创建2D窗口
    
    def __call__(self, img1, img2):
        # 确保图像尺寸匹配
        if img1.shape != img2.shape:
            # 裁剪到相同尺寸
            min_height = min(img1.shape[0], img2.shape[0])
            min_width = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_height, :min_width]
            img2 = img2[:min_height, :min_width]
        
        return _ssim(img1, img2, self.window, 
                    self.window_size, 0, self.size_average)  # channel参数已移除

