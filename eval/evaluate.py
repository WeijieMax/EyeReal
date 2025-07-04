import sys
from lib.NTF.train_NTF import *
from train_eyeReal_fp32 import *
from data.dataset import *
from config.args import get_parser
from config.scene_dict import scene_dict

import math
import os
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms as T
from tqdm import trange
import gc
import time

scene_dict_uco3d = {
    "scale_physical2world":0.28,
    "thickness":6,
    "vertical":"y",
    "ground_coefficient":-0.5,
    "orientation":"xoy"
}

def init_scene_args(args, dataset_name=None):
    "Initialize scene-specific arguments based on dataset name."
    if dataset_name and dataset_name in scene_dict:
        arg_dict = scene_dict[dataset_name]
        print(f"Using scene_dict configuration for dataset: {dataset_name}")
        # arg_dict['scale_physical2world'] = 0.5/6
    else:
        arg_dict = scene_dict_uco3d
        print(f"Using default uco3d configuration for dataset: {dataset_name}")

    args.scale_physical2world = arg_dict["scale_physical2world"]
    args.thickness = arg_dict["thickness"]
    args.vertical = arg_dict["vertical"]
    args.orientation = arg_dict["orientation"]

    if "physical_width" in arg_dict:
        args.physical_width = arg_dict["physical_width"]
    else:
        args.physical_width = 51.84
        
    if "ground_coefficient" in arg_dict:
        args.ground_coefficient = arg_dict["ground_coefficient"]
    else:
        args.ground_coefficient = None
        
    if "ground" in arg_dict:
        args.ground = arg_dict["ground"]
    else:
        args.ground = None
        
    if "delta_x" in arg_dict:
        args.delta_x = arg_dict["delta_x"]
    else:
        args.delta_x = None
        
    if "delta_y" in arg_dict:
        args.delta_y = arg_dict["delta_y"]
    else:
        args.delta_y = None
        
    if "delta_z" in arg_dict:
        args.delta_z = arg_dict["delta_z"]
    else:
        args.delta_z = None


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

def get_screen_coords_world(args, N_screen=3):

    H, W = args.image_height, args.image_width
    
    # 获取场景参数
    if hasattr(args, 'physical_width') and args.physical_width is not None:
        physical_width = args.physical_width
    else:
        physical_width = 51.84
    
    if hasattr(args, 'ground') and args.ground is not None:
        ground = args.ground
    else:
        ground = 0
    
    scale_pixel2world = args.scale_physical2world * physical_width / W
    
    Z_world = args.thickness * args.scale_physical2world
    if hasattr(args, 'ground_coefficient') and args.ground_coefficient is not None:
        z_min = Z_world * args.ground_coefficient
    else:
        z_min = ground
    
    z_max = (z_min + Z_world)

    z_min = z_min - Z_world / 2
    z_max = z_max + Z_world / 2
    
    W_w = W * scale_pixel2world
    H_w = H * scale_pixel2world
    

    delta = torch.tensor([0.0, 0.0, 0.0])
    if hasattr(args, 'delta_x') and args.delta_x is not None:
        delta[0] = args.delta_x
    if hasattr(args, 'delta_y') and args.delta_y is not None:
        delta[1] = args.delta_y
    if hasattr(args, 'delta_z') and args.delta_z is not None:
        delta[2] = args.delta_z
    

    if args.orientation == "xoy":
        coord_screen_world = torch.stack([
            torch.Tensor([[-W_w/2, H_w/2, z], [W_w/2, H_w/2, z], [-W_w/2, -H_w/2, z], [W_w/2, -H_w/2, z]
        ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
    elif args.orientation == "xoz":
        coord_screen_world = torch.stack([
            torch.Tensor([[ -W_w/2,z, H_w/2], [ W_w/2, z,H_w/2], [-W_w/2,z,  -H_w/2], [W_w/2, z, -H_w/2]
        ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
    elif args.orientation == "yox":
        coord_screen_world = torch.stack([
            torch.Tensor([[-H_w/2, -W_w/2, z], [-H_w/2, W_w/2, z], [H_w/2, -W_w/2, z], [H_w/2, W_w/2, z]
        ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
    elif args.orientation == "yoz":
        coord_screen_world = torch.stack([
            torch.Tensor([[z, -W_w/2, H_w/2], [z, W_w/2, H_w/2], [z, -W_w/2, -H_w/2], [z, W_w/2, -H_w/2]
        ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
    elif args.orientation == "zox":
        coord_screen_world = torch.stack([
            torch.Tensor([[H_w/2, z, -W_w/2], [H_w/2, z, W_w/2], [-H_w/2, z, -W_w/2], [-H_w/2, z, W_w/2]
        ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
    else:
        raise ValueError(f"Unsupported orientation: {args.orientation}")


    for index in range(3):
        coord_screen_world[..., index] = coord_screen_world[..., index] + delta[index]
        
    return coord_screen_world

import math
psnr_ls = list()
ssim_ls = list()

def calc_eyeReal(args, model, transform, FOV, data_path, coord_screen_world):
    data_prefix = os.path.join(os.getcwd(), data_path)
    images_path = os.listdir(data_path)
    images_path = sorted(images_path, key=sort_key)
    
    half_num = int(len(images_path)/2)
    
    with torch.no_grad():
        for i in trange(half_num, desc=f"Processing {os.path.basename(data_path)}"):
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

            patterns = model(images, views, coord_screen_world)

            results, masks = model.aggregation(patterns, views, coord_screen_world, FOV=FOV)
            loss = F.mse_loss(results*masks, images*masks)
            psnr = get_PSNR(loss.item(), masks)

            # Skip cases where view direction is parallel to screen
            if math.isnan(psnr):
                continue
            ssim = model.ssim_calc((results*masks).flatten(0,1), (images*masks).flatten(0,1))
            psnr_ls.append(psnr)
            ssim_ls.append(ssim)
            del img0, img1, images, views, patterns, results, masks
            gc.collect()
            torch.cuda.empty_cache()
    
    return psnr_ls, ssim_ls

def extract_dataset_name(dataset_path):

    parent_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(parent_dir)
    return dataset_name

def eval_eyeReal_on_val_dataset(args):
    FOV = 40 / 180 * math.pi
    args.train_NTF = False
    if not args.ckpt_weights:
        raise ValueError("--ckpt_weights must be specified.")
    args.ckpt_weights = args.ckpt_weights
    args.embed_dim=32
    args.N_screen = 3

    if not args.val_root:
        raise ValueError("--val_root must be specified.")
    val_root = args.val_root
    all_datasets = []
    dataset_dirs = os.listdir(val_root)
    
    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(val_root, dataset_dir)
        if os.path.isdir(dataset_path):
            subdirs = os.listdir(dataset_path)
            for subdir in subdirs:
                subdir_path = os.path.join(dataset_path, subdir)
                if os.path.isdir(subdir_path):
                    all_datasets.append(subdir_path)
    
    print(f"Found {len(all_datasets)} validation datasets:")
    for dataset in all_datasets:
        print(f"  - {dataset}")
    print()
    
    results_per_dataset = []
    
    transform = get_transform(args)
    
    for dataset_path in all_datasets:
        print(f"\nEvaluating dataset: {dataset_path}")
        
        dataset_name = extract_dataset_name(dataset_path)
        print(f"Dataset name: {dataset_name}")
        
        init_scene_args(args, dataset_name)
        
        coord_screen_world = get_screen_coords_world(args, N_screen=args.N_screen)
        coord_screen_world = coord_screen_world[None]
        coord_screen_world = coord_screen_world.cuda()
        
        model = EyeRealNet(args=args, FOV=FOV)
        model.cuda()
        
        checkpoint = torch.load(args.ckpt_weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # try:
        psnr, ssim = calc_eyeReal(args=args, model=model, transform=transform, FOV=FOV, 
                                 data_path=dataset_path, coord_screen_world=coord_screen_world)

        del model, coord_screen_world
        gc.collect()
        torch.cuda.empty_cache()
        

    psnr_tensor = torch.tensor(psnr)
    ssim_tensor = torch.tensor(ssim)

    overall_psnr_mean = torch.mean(psnr_tensor)
    overall_psnr_std = torch.std(psnr_tensor)
    overall_ssim_mean = torch.mean(ssim_tensor)
    overall_ssim_std = torch.std(ssim_tensor)
    

    print("\n" + "="*100)
    print("DETAILED RESULTS PER DATASET:")
    print("="*100)
    header = ["Dataset", "Dataset Name", "Config", "PSNR Mean", "PSNR Std", "SSIM Mean", "SSIM Std"]
    print("{:<40} {:<20} {:<15} {:<10} {:<10} {:<10} {:<10}".format(*header))
    print("-" * 105)
    
    for result in results_per_dataset:
        print("{:<40} {:<20} {:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            result['dataset'],
            result['dataset_name'],
            result['scene_config'],
            result['psnr_mean'],
            result['psnr_std'],
            result['ssim_mean'],
            result['ssim_std']
        ))
    

    print("\n" + "="*80)
    print("OVERALL RESULTS FOR ALL VALIDATION DATASETS:")
    print("="*80)
    print(f"Number of datasets evaluated: {len(results_per_dataset)}")
    print(f"Overall PSNR: {overall_psnr_mean:.4f} ± {overall_psnr_std:.4f}")
    print(f"Overall SSIM: {overall_ssim_mean:.4f} ± {overall_ssim_std:.4f}")
    print("="*80)
    
    return {
        'overall_psnr_mean': overall_psnr_mean.item(),
        'overall_psnr_std': overall_psnr_std.item(),
        'overall_ssim_mean': overall_ssim_mean.item(),
        'overall_ssim_std': overall_ssim_std.item(),
        'num_datasets': len(results_per_dataset),
        'per_dataset_results': results_per_dataset
    }


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    

    print("Starting evaluation on validation datasets...")
    results = eval_eyeReal_on_val_dataset(args)
    
    if results:
        print(f"\nFinal Results Summary:")
        print(f"Evaluated {results['num_datasets']} datasets")
        print(f"Overall PSNR: {results['overall_psnr_mean']:.4f} ± {results['overall_psnr_std']:.4f}")
        print(f"Overall SSIM: {results['overall_ssim_mean']:.4f} ± {results['overall_ssim_std']:.4f}")
    else:
        print("Evaluation failed!")