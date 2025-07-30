import random
import math
import torch
import os
from data.render import GaussianRender
from dataset import eye2world_pytroch
from train_EyeReal import init_scene_args
from config.args import get_gaussian_parser

import numpy as np

FOV = 40
R_min, R_max = 80, 150
num = 1000
phi_min, phi_max = 60, 90
theta_min, theta_max = 40, 140 


def randomize(vertical, orientation, scale_physical2world):

    eye_distance = 6*scale_physical2world

    R = scale_physical2world*random.uniform(R_min, R_max)

    phi = math.radians(random.uniform(phi_min, phi_max))
    theta = math.radians(random.uniform(theta_min, theta_max))


    if vertical == "z":
        z1 = z2 = round(R*math.cos(phi), 3)
    elif vertical == "x":
        x1 = x2 = round(R*math.cos(phi), 3)
    elif vertical == "y":
        y1 = y2 = round(R*math.cos(phi), 3)
    else:
        raise ValueError("wrong input vertical")
    

    d = R*math.sin(phi)
    sign = -1 if d <= 0 else 1
    delta = abs(math.atan(0.5*eye_distance/d))
    r = math.sqrt(d**2 + (0.5*eye_distance)**2) * sign
    

    if orientation == "xoy":
        z1 = round(r*math.sin(theta-delta), 3)
        z2 = round(r*math.sin(theta+delta), 3)
        x1 = round(r*math.cos(theta-delta), 3) 
        x2 = round(r*math.cos(theta+delta), 3)
    elif orientation == "xoz":
        y1 = round(r*math.sin(theta-delta), 3)
        y2 = round(r*math.sin(theta+delta), 3)
        x1 = round(r*math.cos(theta-delta), 3) 
        x2 = round(r*math.cos(theta+delta), 3)
    elif orientation == "yox":
        z1 = round(r*math.sin(theta-delta), 3)
        z2 = round(r*math.sin(theta+delta), 3)
        y1 = round(r*math.cos(theta-delta), 3) 
        y2 = round(r*math.cos(theta+delta), 3)
    elif orientation == "yoz":
        x1 = round(r*math.sin(theta-delta), 3)
        x2 = round(r*math.sin(theta+delta), 3)
        y1 = round(r*math.cos(theta-delta), 3) 
        y2 = round(r*math.cos(theta+delta), 3)
    elif orientation == "zox":
        y1 = round(r*math.sin(theta-delta), 3)
        y2 = round(r*math.sin(theta+delta), 3)
        z1 = round(r*math.cos(theta-delta), 3) 
        z2 = round(r*math.cos(theta+delta), 3)
    (x1, y1, z1), (x2, y2, z2) = (x1, y1, z1), (x2, y2, z2)

    return (x1, y1, z1), (x2, y2, z2)

if __name__ == "__main__":

    root = './weight/gaussian_ply'
    for gs in os.listdir(root):
        gaussian_path = os.path.join(root, gs)
        render = GaussianRender(
            parser=get_gaussian_parser(),
            sh_degree=3, 
            gaussians_path=gaussian_path,
            white_background=True, 
            FOV=FOV / 180 * math.pi)

        scene_name = gs.split('.')[0]
        

        from config.args import get_parser
        parser = get_parser()
        args = parser.parse_args()
        args.scene = scene_name
        init_scene_args(args=args)


        file_path = './dataset/scene_data_no_disrupt/{}/'.format(args.scene)

        if 'museum' in scene_name or 'room_floor' in scene_name:
            num = 1500
        else:
            num = 500

        data_folder = '{}{}_scale_{}_R_{}_{}_FOV_{}_theta_{}_{}_phi_{}_{}'.format(
            args.scene, num, round(args.scale_physical2world,3), 
            R_min, R_max, FOV, theta_min, theta_max, phi_min, phi_max)
        data_path = os.path.join(file_path, data_folder)
        os.makedirs(data_path, exist_ok=True)


        from tqdm import tqdm
        import torchvision
        for i in tqdm(range(num)):

            eye1, eye2 = randomize(args.vertical, args.orientation, args.scale_physical2world)
            eye1_t = torch.tensor(eye1)
            eye2_t = torch.tensor(eye2)
            

            eye1_view = eye2world_pytroch(args.vertical, eye1_t).cuda()
            eye1_img = render.render_from_view(eye1_view)
            torchvision.utils.save_image(
                eye1_img, 
                data_path + "/" + 'pair{}_x{}_y{}_z{}_{}'.format(
                    i, round(eye1[0], 3), round(eye1[1], 3), round(eye1[2], 3), 0) + ".jpg")
            

            eye2_view = eye2world_pytroch(args.vertical, eye2_t).cuda()
            eye2_img = render.render_from_view(eye2_view)
            torchvision.utils.save_image(
                eye2_img, 
                data_path + "/" + 'pair{}_x{}_y{}_z{}_{}'.format(
                    i, round(eye2[0], 3), round(eye2[1], 3), round(eye2[2], 3), 1) + ".jpg")
        


    
