import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.transforms.functional import perspective
import math
from torchvision import transforms as T
import os
from PIL import Image
import model.metric as pytorch_ssim

def get_transform(image_size):
    transforms = [
        T.Resize((image_size[0], image_size[1])),
        T.ToTensor(),
        # T.Normalize(
        #     mean=[0.485, 0.456, 0.406], 
        #     std=[0.229, 0.224, 0.225]),
    ]

    return T.Compose(transforms)

class DisplayNetNTF():
    def __init__(self, pattern_size, num_side, view_ratio=(1, 1), device='cpu'):
        super(DisplayNetNTF, self).__init__()
        self.pattern_size = torch.Size(pattern_size) # 3 H W
        self.device = device

        self.rear = torch.rand(self.pattern_size).to(device=device)
        self.middle = torch.rand(self.pattern_size).to(device=device)
        self.front = torch.rand(self.pattern_size).to(device=device)
        self.view_ratio = view_ratio
        self.view = self.views_generation(num_side).to(device=device)
        self.bias = torch.full(self.pattern_size, 1e-5, dtype=torch.float32).to(device=device)
        self.ssim_calc = pytorch_ssim.SSIM()
    
    def views_generation(self, num_side, step=1):
        # assert (num_side-1) % 2 == 0
        # n = int((num_side-1)/2)
        # base = torch.arange(-n, n+1, step=step)
        # view_H = base[:, None].repeat(1, num_side).view(-1, 1)
        # view_W = base.repeat(num_side, 1).view(-1, 1)
        
        # return torch.cat([view_H, view_W], dim=-1)
        return torch.tensor([
                    [0, -0.5],
                    [0, 0.5]
                ])

    def mask_generation(self, view):
        H, W = self.middle.shape[-2:]
        h, w = view
        h_abs = int(round(abs(h)))
        
        w_abs = int(round(abs(w)))
        mask = torch.ones(H, W).to(device=self.device)
        if h >= 0:
            mask[:h_abs, :] = 0
        else:
            mask[H-h_abs:, :] = 0
        if w >= 0:
            mask[:, :w_abs] = 0
        else:
            mask[:, W-w_abs:] = 0

        return mask
    
    def refineLayer(self, views, images, refine_layer=None):
        
        numerator = torch.zeros(self.pattern_size, dtype=torch.float32).to(device=self.device)
        denominator = torch.zeros(self.pattern_size, dtype=torch.float32).to(device=self.device)
        for i, view_ in enumerate(views):
            view = torch.stack([view_[0] * self.view_ratio[0], 
                                view_[1] * self.view_ratio[1]]).to(device=self.device)
            
            view_shifts = (view[0].round().int().item(), view[1].round().int().item()) 
            anti_view_shifts = (-view[0].round().int().item(), -view[1].round().int().item()) 

            add_result = (torch.roll(self.front, shifts=view_shifts, dims=(-2, -1)) * self.mask_generation(view_shifts)) + \
                            self.middle + \
                        (torch.roll(self.rear, shifts=anti_view_shifts, dims=(-2, -1)) * self.mask_generation(anti_view_shifts))
            #padsize+phi，3*padsize-phi都有些古怪，和我们的计数规则不一样
            if refine_layer == 'rear':
                numerator += torch.roll(images[i], shifts=view_shifts, dims=(-2, -1)) * self.mask_generation(view_shifts) 
                denominator += torch.roll(add_result, shifts=view_shifts, dims=(-2, -1))  * self.mask_generation(view_shifts)
            elif refine_layer == 'middle':
                numerator += images[i]
                denominator += add_result 
            elif refine_layer == 'front':
                numerator += torch.roll(images[i], shifts=anti_view_shifts, dims=(-2, -1)) * self.mask_generation(anti_view_shifts)  
                denominator += torch.roll(add_result, shifts=anti_view_shifts, dims=(-2, -1)) * self.mask_generation(anti_view_shifts) 
            else:
                raise ValueError("refine_layer not formal")

        if refine_layer == 'rear':
            self.rear = self.rear * (numerator + self.bias) / (denominator + self.bias) 
            self.rear = torch.clamp(self.rear, 0, 1)
        elif refine_layer == 'middle':
            self.middle = self.middle * (numerator + self.bias) / (denominator + self.bias) 
            self.middle = torch.clamp(self.middle, 0, 1)
        elif refine_layer == 'front':
            self.front = self.front * (numerator + self.bias) / (denominator + self.bias) 
            self.front = torch.clamp(self.front, 0, 1)
        else:
            raise ValueError("refine_layer not formal")
    def update(self, views, images):
        self.refineLayer(views, images, "front")
        self.refineLayer(views, images, "middle")
        self.refineLayer(views, images, "rear")

    def getResults(self, views, images):
        add_results = list()
        res_masks = list()
        for i, view_ in enumerate(views):
            view = torch.stack([view_[0] * self.view_ratio[0], 
                                view_[1] * self.view_ratio[1]]).to(device=self.device)
            view_shifts = (view[0].round().int().item(), view[1].round().int().item()) 
            anti_view_shifts = (-view[0].round().int().item(), -view[1].round().int().item()) 
            add_result = (torch.roll(self.front, shifts=view_shifts, dims=(-2, -1)) * self.mask_generation(view_shifts)) + \
                            self.middle + \
                        (torch.roll(self.rear, shifts=anti_view_shifts, dims=(-2, -1)) * self.mask_generation(anti_view_shifts))

            add_results.append(add_result)
            res_masks.append((self.mask_generation(view_shifts) * self.mask_generation(anti_view_shifts))[None])
        return torch.stack(add_results), torch.stack(res_masks)
    @staticmethod
    def get_PSNR(MSE, mask: torch.Tensor=None, MAX=1):
        if mask is not None:
            MSE = MSE*mask.nelement()/mask.sum()
        return 10*math.log10(MAX**2/MSE)


        