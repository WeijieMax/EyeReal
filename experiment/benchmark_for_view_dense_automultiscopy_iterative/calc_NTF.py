import os
import sys
import math
import time
import argparse
from tqdm import tqdm
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from displaynet_NTF import DisplayNetNTF
import torch.nn.functional as F
import gc
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import wandb
from data.render import GaussianRender
from config.args import get_gaussian_parser
def sort_key(s):
    # "pair0_x5.896_y45.396_z-24.465_left.jpg"
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)

def get_dataset(transform, args):
    from dataset_NTF import NTFDataset

    data_prefix = os.path.join(os.getcwd(), args.data_path)
    images_path = os.listdir(args.data_path)
    images_path = sorted(images_path, key=sort_key)


    ds = NTFDataset(images_path=images_path,
                        data_prefix=data_prefix,
                        transform=transform
                       )
    return ds

def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
    ]
    return T.Compose(transforms)
from torchvision.transforms.functional import perspective
def view_tranform(imgs, view, coord_src, coord_src_img, FOV=40/180*math.pi):
        N, _, H, W = imgs.shape
        fx = W/2 / math.tan(FOV/2)
        # import pdb;pdb.set_trace()
        coord_src_homo = torch.cat([coord_src.cpu(), torch.ones(N,4,1)], dim=-1).to(imgs.device)
        # coord_dst = torch.matmul(torch.inverse(view)[:, None], coord_src_homo[..., None]).squeeze(-1)[..., :3] # N 4 3
        coord_dst = torch.matmul(torch.inverse(view.float())[:, None], coord_src_homo.float()[..., None]).squeeze(-1)[..., :3] # N 4 3
        u = (-fx*coord_dst[..., [0]]/coord_dst[..., [2]] + W/2)
        v = (fx*coord_dst[..., [1]]/coord_dst[..., [2]] + H/2)
        coord_dst_img = torch.cat([u, v], dim=-1)

        masks = torch.ones_like(imgs)

        masks_new = torch.stack([perspective(mask, src.tolist(), dst.tolist()) 
                            for mask, src, dst in zip(masks, coord_src_img, coord_dst_img)])

        return masks_new
    
def get_masks(imgs=None, views=None):
        """
        imgs: (N_in, C_rgb, H, W)
        views: (B, N_in,  4, 4)
        coord_screen_world: (N_s, 4, 3)
        coord_pixel_init:   (N_s, 4, 2)
        返回
        masks: (N_in , C_rgb H, W)   
        """

        N_in, C_rgb, H, W = imgs.shape

 
        masks_new = view_tranform(
            imgs.view(1, N_in, C_rgb, H, W).repeat(1, N_s, 1, 1, 1).flatten(0, 1),
            views.view(1, N_in, 4, 4).repeat(N_s, 1, 1, 1).flatten(0, 1),
            coord_screen_world.view( N_s, 1, 4, 3).repeat(1, N_in, 1, 1).flatten(0, 1),
            coord_pixel_init.view(N_s, 1, 4, 2).repeat(1, N_in, 1, 1).flatten(0, 1),
        )
            

        masks_new = masks_new.reshape(N_in, N_s, C_rgb, H, W)
        # 在 N_s 维度上做乘积（prod），得到 (N_in, H, W, C_rgb)
        masks = masks_new.prod(dim=1)

        return masks
def get_screen_coords_world(thickness, scale_physical2world, physical_width, ground, ground_coefficient, orientation, delta):
        '''
        thickness: the physical length of whole thickness between screens at both sides
        '''
        
        H, W = 1080, 1920
        N_screen = 3
        
        if physical_width == None:
            physical_width = 51.84
        if ground == None:
            ground = 0
        
        scale_pixel2world = scale_physical2world * physical_width / W
        
        Z_world = thickness * scale_physical2world
        if ground_coefficient != None:
            z_min = Z_world * ground_coefficient
        else:
            z_min = ground
            
        z_max = (z_min + Z_world)
        W_w = W * scale_pixel2world
        H_w = H * scale_pixel2world
        
        if orientation == "xoy":
            coord_screen_world = torch.stack([
                torch.Tensor([[-W_w/2, H_w/2, z], [W_w/2, H_w/2, z], [-W_w/2, -H_w/2, z], [W_w/2, -H_w/2, z]
            ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
        elif orientation == "xoz":
            coord_screen_world = torch.stack([
                torch.Tensor([[ -W_w/2,z, H_w/2], [ W_w/2, z,H_w/2], [-W_w/2,z,  -H_w/2], [W_w/2, z, -H_w/2]
            ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
        elif orientation == "yox":
            coord_screen_world = torch.stack([
                torch.Tensor([[-H_w/2, -W_w/2, z], [-H_w/2, W_w/2, z], [H_w/2, -W_w/2, z], [H_w/2, W_w/2, z]
            ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
        elif orientation == "yoz":
            coord_screen_world = torch.stack([
                torch.Tensor([[z, -W_w/2, H_w/2], [z, W_w/2, H_w/2], [z, -W_w/2, -H_w/2], [z, W_w/2, -H_w/2]
            ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])
        elif orientation == "zox":
            coord_screen_world = torch.stack([
                torch.Tensor([[H_w/2, z, -W_w/2], [H_w/2, z, W_w/2], [-H_w/2, z, -W_w/2], [-H_w/2, z, W_w/2]
            ]) for z in torch.linspace(z_min, z_max, N_screen).tolist()])


        for index in range(3):
            coord_screen_world[..., index] = coord_screen_world[..., index] + delta[index]
            
        return coord_screen_world
from config.scene_dict import *
def init_scene_args(args):

    if args.scene in scene_dict:
        arg_dict = scene_dict[args.scene]
    else:
        # arg_dict = scene_dict_uco3d
        arg_dict["scale_physical2world"] = 0.28
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

    print(arg_dict)

    delta = torch.tensor([0, 0, 0])
    delta[0] = arg_dict.get('delta_x') if arg_dict.get('delta_x') else 0
    delta[1] = arg_dict.get('delta_y') if arg_dict.get('delta_x') else 0
    delta[2] = arg_dict.get('delta_z') if arg_dict.get('delta_x') else 0
    # import pdb;pdb.set_trace()
    coord_screen_world = get_screen_coords_world(
        thickness = arg_dict.get('thickness'), 
        scale_physical2world = arg_dict.get('scale_physical2world'), 
        physical_width = arg_dict.get('physical_width'), 
        ground = arg_dict.get('ground'), 
        ground_coefficient = arg_dict.get('ground_coefficient'), 
        orientation = arg_dict.get('orientation'), 
        delta = delta
    )

    return coord_screen_world
def update_two_views(model:DisplayNetNTF, iteration, data, device, output_path):
    with torch.no_grad():
        render = GaussianRender(
            parser=get_gaussian_parser(),
            sh_degree=3, 
            gaussians_path=r"./weight/gaussian_ply/lego_bulldozer.ply",
            white_background=True, FOV=40 / 180 * math.pi, render_image_size=(1080,1920))
        exe_time = 0

        images_, views, mask_views = data
        views, mask_views = views.to(device), mask_views.to(device)

        start_time1 = time.time()
        images = torch.stack([render.render_from_view(mask_view) for mask_view in mask_views])
        exe_time += time.time() - start_time1

    for i in range(iteration):
        start_time2 = time.time()
        model.update(views, images)
        exe_time += time.time() - start_time2


    results, _ = model.getResults(views, images)
    masks = get_masks(images, mask_views)
    
    psnr = model.get_PSNR(F.mse_loss(results*masks, images*masks))
    ssim = model.ssim_calc(results*masks, images*masks)

    # -----释放本地变量-----
    del images_, views, mask_views, images, results, masks, render
    gc.collect()                 # Python显式垃圾回收
    torch.cuda.empty_cache()     # 释放PyTorch空闲显存

    return psnr,ssim,exe_time

from model.network import EyeRealNet
def main(args):

    dataset = get_dataset(transform=get_transform(args), args=args)
    exp_name = args.exp_name
    device = 'cuda:0'

    output_path = args.output_dir + exp_name
    os.makedirs(output_path, exist_ok=True)

    view_ratio_ = 1920 / 518.4 * 60
    model = DisplayNetNTF(pattern_size=(3, args.image_height, args.image_width), num_side=2, 
                                  view_ratio=(view_ratio_, view_ratio_), device=device) # 593/64 840/64
    
    psnr_ls = list()
    ssim_ls = list()
    time_ls = list()
    all_num = args.paper_all_num
    iteration = args.paper_iteration
    pbar = tqdm(range(all_num), desc="Processing", unit="iteration")
    for dataidx in pbar:
        # train
        data = dataset.__getitem__(dataidx)
        psnr, ssim, time = update_two_views(model=model, iteration=iteration, 
                                            data=data, device=device, output_path=output_path)
        psnr_ls.append(psnr)
        ssim_ls.append(ssim)
        time_ls.append(time)
        pbar.set_postfix(PSNR=f"{psnr:.4f}", SSIM=f"{ssim:.4f}", Time=f"{time:.4f}s")

    psnr_t = torch.tensor(psnr_ls)
    ssim_t = torch.tensor(ssim_ls)
    time_t = torch.tensor(time_ls)

    torch.save(psnr_t, output_path + "/" + "{}_psnr.pt".format(exp_name))
    torch.save(ssim_t, output_path + "/" + "{}_ssim.pt".format(exp_name))
    torch.save(time_t, output_path + "/" + "{}_time.pt".format(exp_name))

    with open('{}/output.txt'.format(output_path), 'a') as f:
        f.write("{}_all_psnr: {}\n".format( exp_name, torch.sum(psnr_t)))
        f.write("{}_all_ssim: {}\n".format( exp_name, torch.sum(ssim_t)))
        f.write("{}_all_time: {}\n".format( exp_name, torch.sum(time_t)))
        f.write("{}_avg_psnr: {}\n".format( exp_name, torch.mean(psnr_t)))
        f.write("{}_avg_ssim: {}\n".format( exp_name, torch.mean(ssim_t)))
        f.write("{}_avg_time: {}\n".format( exp_name, torch.mean(time_t)))
        f.write("{}_stddev_psnr: {}\n".format( exp_name, torch.std(psnr_t)))
        f.write("{}_stddev_ssim: {}\n".format( exp_name, torch.std(ssim_t)))
        f.write("{}_stddev_time: {}\n".format( exp_name, torch.std(time_t)))
        



from tqdm import trange
from config.args import get_parser
W,H = 1920,1080
N_s = 3
parser = get_parser()
args = parser.parse_args()

args.scene = 'lego_bulldozer'
args.data_path = ''
args.output_dir = "./outputs/experiment/NTF/"
args.image_height = 1080
args.image_width = 1920
args.exp_name = ""
args.paper_iteration = 50
args.paper_all_num = 200
coord_screen_world = init_scene_args(args=args)
coord_pixel_init = torch.Tensor([(0, 0), (W, 0), (0, H), (W, H)]).view(1,4,2).repeat(N_s,1,1)
for iter in [5, 25, 50, 125]:
    for start in range(10, 150, 20):
        args.paper_iteration = iter
        args.data_path = r"dataset\eval\lego_bulldozer\lego_bulldozer200_scale_0.083_R_{}_{}_FOV_40_theta_40_140_phi_60_120".format(start, start+20)
        args.exp_name = "data200_R_{}_{}_iter{}".format(start, start+20, iter)
        main(args)