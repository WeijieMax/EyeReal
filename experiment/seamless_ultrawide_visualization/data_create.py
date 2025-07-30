import random
import math
import torch
import os
from data.render import GaussianRender
FOV = 40
R_min, R_max = 0, 400
num = 40000
phi_min, phi_max = math.radians(40), math.radians(140)
theta_min, theta_max = math.radians(40), math.radians(140)
scale = 0.5/6
eye_distance = 6*scale


def sample():

    while True:

        x = random.uniform(R_min, R_max)
        y = random.uniform(-R_max*math.sin(phi_max/2), R_max*math.sin(phi_max/2))
        z = random.uniform(-R_max*math.sin(phi_max/2), R_max*math.sin(phi_max/2))

        r = torch.sqrt(torch.tensor(x**2+y**2+z**2))
        phi = torch.arccos(z / r)
        d = r*torch.sin(phi)
        theta = torch.arccos(y / d)
        if (R_min <= r and r <= R_max and phi_min <= phi and phi <= phi_max and theta_min <= theta and  theta <= theta_max):
            return randomize((r*scale).item(), phi.item(), theta.item())


def randomize(R, phi, theta):

    z = round(R*math.cos(phi), 3)
    
    d = R*math.sin(phi)
    delta = math.atan(0.5*eye_distance/d)
    r = math.sqrt(d**2 + (0.5*eye_distance)**2)
    x1 = round(r*math.sin(theta-delta), 3) # x-z exchange
    y1 = round(r*math.cos(theta-delta), 3)
    x2 = round(r*math.sin(theta+delta), 3)
    y2 = round(r*math.cos(theta+delta), 3)

    return (x1, y1, z), (x2, y2, z)

def eye2world_pytroch(eye_world: torch.Tensor):
    eye_world = eye_world.float()

    vecz = eye_world 
    vecz = vecz / torch.linalg.norm(vecz)

    vecz_w = torch.tensor([1e-4, 1e-5, 1.0]).to(eye_world.device)

    vecx = torch.cross(vecz_w, vecz)
    vecx = vecx / torch.linalg.norm(vecx)

    vecy = torch.cross(vecz, vecx)
    vecy = vecy / torch.linalg.norm(vecy)

    rot = torch.stack([vecx, vecy, vecz]).T

    rt = torch.eye(4).to(eye_world.device)
    rt[:3, :3] = rot
    rt[:3, 3] = eye_world
 
    return rt

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='Auto3D training and testing')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--Z_scale', type=float, default=0.5)
    parser.add_argument('--N_screen', type=int, default=3)
    parser.add_argument('--N_input', type=int, default=2)
    parser.add_argument("--input_size", type=tuple, default=(1080, 1920))
    parser.add_argument("--image_size", type=tuple, default=(1080, 1920))
    parser.add_argument("--sin", action='store_false')
    parser.add_argument("--arcsin", action='store_true')
    parser.add_argument("--ssim", action='store_true')
    parser.add_argument("--add_ssim", action='store_false')
    parser.add_argument("--mul", action='store_true')
    parser.add_argument("--data_aug", action='store_true')
    parser.add_argument("--muladd", action='store_true')
    parser.add_argument("--blender", action='store_false')
    parser.add_argument("--save_preds", action='store_false')
    parser.add_argument("--sigmoid", action='store_false')
    parser.add_argument("--sigmoid_value", type=float, default=0)
    parser.add_argument("--embed_dim", type=int, default=80)
    parser.add_argument('--data_path', default='./data/lego')
    parser.add_argument('--multi_train', action='store_true')

    return parser

if __name__ == "__main__":

    render = GaussianRender(
        parser=get_parser(),
        sh_degree=3, 
        gaussians_path=r'./weight/gaussian_ply/lego_bulldozer.ply',
        white_background=True, FOV=FOV / 180 * math.pi)
    

    file_path = './outputs/experiment/3D-distribute'

    data_folder = r'lego_bulldozer{}_scale_{}_R_{}_{}_FOV_{}_theta_{}_phi_{}'.format(num, round(scale,3), R_min, R_max, FOV, round(theta_max/2, 2), round(phi_max/2, 2))
    data_path = os.path.join(file_path, data_folder)
    os.makedirs(data_path, exist_ok=True)

    
    from tqdm import tqdm
    import torchvision
    for i in tqdm(range(num)):
        eye1, eye2 = sample()
        eye1_t = torch.tensor(eye1)
        eye2_t = torch.tensor(eye2)
        eye1_view = eye2world_pytroch(eye1_t).cuda()
        eye1_img = render.render_from_view(eye1_view)
        torchvision.utils.save_image(eye1_img, data_path + "/" + 'pair{}_x{}_y{}_z{}_{}'.format(i, round(eye1[0], 3), round(eye1[1], 3), round(eye1[2], 3), 0) + ".jpg")
        eye2_view = eye2world_pytroch(eye2_t).cuda()
        eye2_img = render.render_from_view(eye2_view)
        torchvision.utils.save_image(eye2_img, data_path + "/" + 'pair{}_x{}_y{}_z{}_{}'.format(i, round(eye2[0], 3), round(eye2[1], 3), round(eye2[2], 3), 1) + ".jpg")



    
