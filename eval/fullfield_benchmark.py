import sys
from lib.NTF.train_NTF import *
from train_eyeReal_fp32 import *
from data.dataset import *
from config.args import get_parser

import math
import os
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms as T
from tqdm import trange
import gc
import time
from config.scene_dict import *


def init_scene_args(args):

    if args.scene in scene_dict:
        arg_dict = scene_dict[args.scene]

        args.scale_physical2world = arg_dict["scale_physical2world"]
        args.thickness = arg_dict["thickness"]
        args.vertical = arg_dict["vertical"]
        args.orientation = arg_dict["orientation"]

        if "physical_width" in arg_dict:
            args.physical_width = arg_dict["physical_width"]
        if "ground_coefficient" in arg_dict:
            args.ground_coefficient = arg_dict["ground_coefficient"]
        if "ground" in arg_dict:
            args.ground = arg_dict["ground"]
        if "delta_x" in arg_dict:
            args.delta_x = arg_dict["delta_x"]
        if "delta_y" in arg_dict:
            args.delta_y = arg_dict["delta_y"]
        if "delta_z" in arg_dict:
            args.delta_z = arg_dict["delta_z"]
    else:
        raise ValueError("wrong input scene name")


def eval_ntf(args):
    args.train_NTF = True
    psnr_res = []
    ssim_res = []
    
    for iter in [5, 25, 50]:
        for start in range(10, 150, 20):
            args.ntf_iteration = iter
            args.data_path = "dataset/eval/lego_bulldozer/lego_bulldozer200_scale_0.083_R_{}_{}_FOV_40_theta_40_140_phi_60_120".format(start, start+20)
            psnr, ssim = train_ntf(args)
            psnr_res.append(psnr)
            ssim_res.append(ssim)
    header = ["R", "iter", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"]
    print("ntf eval:")
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*header))
    index = 0
    for iter in [5, 25, 50]:
        for start in range(10, 150, 20):
            print("{:<10} {:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format("{}-{}".format(start, start+20), "{}".format(iter),  psnr_res[index][0], 
                                                                                              psnr_res[index][1], ssim_res[index][0], ssim_res[index][1]))
            index += 1


def sort_key(s):
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)


def get_ori_coord(s):
    pairs = s.split('_')

    x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
    return (x,y,z)
   
def get_img_matrix(s, vertical):
    ori_x, ori_y, ori_z = get_ori_coord(s)
    eye_world = torch.tensor([ori_x, ori_y, ori_z])
    return eye2world_pytroch(vertical=vertical, eye_world=eye_world)

def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
    ]

    return T.Compose(transforms)



def calc_eyeReal(args, model, transform, FOV):
    
    data_prefix = os.path.join(os.getcwd(), args.data_path)
    images_path = os.listdir(args.data_path)
    images_path = sorted(images_path, key=sort_key)
    
    half_num = int(len(images_path)/2)
    psnr_ls = list()
    ssim_ls = list()
    with torch.no_grad():
        for i in trange(half_num):
            i_0 = i * 2
            i_1 = i * 2 + 1
            img0 = Image.open(os.path.join(data_prefix, images_path[i_0]))
            img0 = transform(img0)
            view0 = get_img_matrix(images_path[i_0], args.vertical)
            img1 = Image.open(os.path.join(data_prefix, images_path[i_1]))
            img1 = transform(img1)
            view1 = get_img_matrix(images_path[i_1], args.vertical)
            images = torch.stack([img0, img1], dim=0)[None]
            views = torch.stack([view0, view1], dim=0)[None]
            images, views = images.cuda(non_blocking=True), views.cuda(non_blocking=True)

            patterns = model(images, views)

            results, masks = model.aggregation(patterns, views, FOV)
            loss = F.mse_loss(results*masks, images*masks)
            psnr = get_PSNR(loss.item(), masks)
            ssim = model.ssim_calc((results*masks).flatten(0,1), (images*masks).flatten(0,1))
            psnr_ls.append(psnr)
            ssim_ls.append(ssim)
            del img0, img1, images, views, patterns, results, masks
            gc.collect()
            torch.cuda.empty_cache()
    psnr_t = torch.tensor(psnr_ls)
    ssim_t = torch.tensor(ssim_ls)
    
    return  (torch.mean(psnr_t), torch.std(psnr_t)),\
            (torch.mean(ssim_t), torch.std(ssim_t))

def eval_eyeReal(args):
    FOV = 40 / 180 * math.pi
    args.scene = 'lego_bulldozer'
    args.train_NTF = False
    args.ckpt_weights = '/fs-computility/ai4chem/maweijie/eyemulti/outputs/data30+17_mutex-lr0.005-ep20-2025-06-27-15:31:13/model_best_10_train.pth'
    args.embed_dim=32
    args.N_screen = 3

    init_scene_args(args)
    args.scale_physical2world=0.5/6
    transform = get_transform(args)
    model = EyeRealNet(args=args, FOV=FOV)
    model.cuda()
    checkpoint = torch.load(args.ckpt_weights, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    psnr_res = []
    ssim_res = []
    for start in range(10, 150, 20):
        
        args.data_path = "/fs-computility/ai4chem/maweijie/Eyereal_paper/lego_bulldozer"
        psnr, ssim = calc_eyeReal(args=args, model=model, transform=transform, FOV=FOV)
        psnr_res.append(psnr)
        ssim_res.append(ssim)
    header = ["R", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"]
    print("eyeReal eval:")
    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(*header))
    index = 0
    for start in range(10, 150, 20):
        print("{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format("{}-{}".format(start, start+20), psnr_res[index][0], psnr_res[index][1], 
                                                                                          ssim_res[index][0], ssim_res[index][1]))
        index += 1

if __name__ == '__main__':
    
    

    parser = get_parser()
    args = parser.parse_args()
    # eval_ntf(args)
    eval_eyeReal(args)