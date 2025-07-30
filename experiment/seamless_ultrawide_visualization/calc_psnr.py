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
from model.metric import *
def get_screen_coords_world(size, thickness, scale_physical2world, physical_width, ground, ground_coefficient, orientation, delta):
    '''
    thickness: the physical length of whole thickness between screens at both sides
    '''
    
    H, W = size
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
###########################
# torch.cuda.set_device(0)
parser = get_parser()
args = parser.parse_args()
args.scene = 'lego_bulldozer'
args.FOV = 40
# args.embed_dim = 16
args.embed_dim = 32
args.model_choice = 0
init_scene_args(args=args)


#################################################################################
def sort_key(s):
    # "pair0_x5.896_y45.396_z-24.465_left.jpg"
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)




#####################################################################

FOV = args.FOV
if FOV > math.pi:
    FOV = FOV / 180 * math.pi


################################################################################
def eye2world_pytroch(eye_world: torch.Tensor):
    eye_world = eye_world.float()
    vecz = eye_world
    vecz = vecz / torch.linalg.norm(vecz)
    vecz_w = torch.tensor([1e-5, 1e-6, 1.]).to(eye_world.device)
    vecx = torch.cross(vecz_w, vecz)
    vecx = vecx / torch.linalg.norm(vecx)
    vecy = torch.cross(vecz, vecx)
    vecy = vecy / torch.linalg.norm(vecy)
    rot = torch.stack([vecx, vecy, vecz]).T
    rt = torch.eye(4).to(eye_world.device)
    rt[:3, :3] = rot
    rt[:3, 3] = eye_world
    
    return rt

def get_ori_coord( s):
    # "pair0_x5.896_y45.396_z-24.465_left"
    pairs = s.split('_')

    x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
    return (x,y,z)
   
def get_img_matrix( s):
    ori_x, ori_y, ori_z = get_ori_coord(s)
    eye_world = torch.tensor([ori_x, ori_y, ori_z])
    return eye2world_pytroch(eye_world), eye_world
######################
def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
    ]

    return T.Compose(transforms)

transform = get_transform(args)
coord_screen_world = get_screen_coords_world(
        size=(1080, 1920),
        thickness=args.thickness,
        scale_physical2world=args.scale_physical2world,
        physical_width=args.physical_width,
        ground=args.ground,
        ground_coefficient=args.ground_coefficient,
        orientation=args.orientation,
        delta=torch.tensor([args.delta_x, args.delta_y, args.delta_z])
    )
from model.network import EyeRealNet
def load_ckpt_calc(args, ckpt):

    data_prefix = os.path.join(os.getcwd(), args.data_path)
    images_path = os.listdir(args.data_path)
    images_path = sorted(images_path, key=sort_key)

    model = EyeRealNet(args=args, FOV=FOV)
    model.cuda()
    checkpoint = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    half_num = int(len(images_path)/2)
    psnr_ls = list() 
    ssim_ls = list()
    world_ls = list()
    coord_screen_world_ = coord_screen_world[None].cuda()
    with torch.no_grad():
        for i in trange(half_num):
            i_0 = i * 2
            i_1 = i * 2 + 1
            img0 = Image.open(os.path.join(data_prefix, images_path[i_0]))
            img0 = transform(img0)
            view0, eye0world = get_img_matrix(images_path[i_0])
            img1 = Image.open(os.path.join(data_prefix, images_path[i_1]))
            img1 = transform(img1)
            view1, eye1world = get_img_matrix(images_path[i_1])
            images = torch.stack([img0, img1], dim=0)[None]
            views = torch.stack([view0, view1], dim=0)[None]
            images, views = images.cuda(non_blocking=True), views.cuda(non_blocking=True)
            patterns = model(images, views, coord_screen_world_)
            results, masks = model.aggregation(patterns, views, coord_screen_world_, FOV)
            loss = F.mse_loss(results*masks, images*masks)
            psnr = get_PSNR(loss.item(), masks)
            ssim = model.ssim_calc((results*masks).flatten(0,1), (images*masks).flatten(0,1))
            psnr_ls.append(psnr)
            ssim_ls.append(ssim)
            world_ls.append((eye0world+eye1world)/2)
            del img0, img1, images, views, patterns, results, masks
            gc.collect()
            torch.cuda.empty_cache()
    psnr_t = torch.tensor(psnr_ls)
    ssim_t = torch.tensor(ssim_ls)
    world_t = torch.stack(world_ls)

    exp_name = 'your_exp_name'
    output_path = './outputs/experiment/3D-distribute/' + exp_name
    os.makedirs(output_path, exist_ok=True)
    torch.save(ssim_t, output_path + "/" + "{}_ssim.pt".format(exp_name))
    torch.save(psnr_t, output_path + "/" + "{}_psnr.pt".format(exp_name))
    torch.save(world_t, output_path + "/" + "{}_worldCoord.pt".format(exp_name))

    with open('{}/output.txt'.format(output_path), 'a') as f:
        f.write("{}_avg_psnr: {}\n".format( exp_name, torch.mean(psnr_t)))
        f.write("{}_avg_ssim: {}\n".format( exp_name, torch.mean(ssim_t)))


    ckpt = r'./weight/model_ckpts/pretrained_model.pth'
args.data_path = "./outputs/experiment/3D-distribute/lego_bulldozer40000_scale_0.083_R_0_400_FOV_40_theta_1.22_phi_1.22"
args.N_screen = 3
load_ckpt_calc(args, ckpt)