import cv2
import numpy as np
import os
import torch
import math
import json
from torchvision.transforms.functional import to_pil_image
import torchvision
import warnings
from copy import deepcopy as c
import random
from config.args import get_parser
from train import init_scene_args
from model.network import EyeRealNet
from data.dataset import eye2world_pytroch
warnings.filterwarnings('ignore')

def randomize(R, phi, theta, vertical, orientation, scale_physical2world):

    eye_distance = 6*scale_physical2world
    R = R*scale_physical2world
    phi = math.radians(phi)
    theta = math.radians(theta)

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

    return (x1, y1, z1), (x2, y2, z2)




args = get_parser().parse_args()
args.scene = 'lego_bulldozer'
init_scene_args(args=args)


model = EyeRealNet(args=args, FOV=40/180*math.pi)
model.load_state_dict(torch.load(r'./weight/model_ckpts/lego_bulldozer.pth', map_location='cpu')['model'])
model = model.cuda()
model.eval()

phi_ls = [70, 80, 90]
theta_ls = [80, 90, 100]
R_ls = [55, 65, 75]

data_path = f"./outputs/calibration/{args.scene}/"
os.makedirs(data_path, exist_ok=True)
# scale = 24 * math.tan(math.radians(30)) * 2 / 120.96
for R in R_ls:
    folder_path = data_path + "R_{}/".format(R)
    os.makedirs(folder_path, exist_ok=True)
    for phi in phi_ls:
        for theta in theta_ls:
            # phi, theta = 70, 90
            eye_right, eye_left = randomize(R = R, phi = phi, theta = theta, 
                                            vertical=args.vertical, orientation=args.orientation, scale_physical2world=args.scale_physical2world)
            eye_right_t = torch.tensor(eye_right)
            eye_left_t = torch.tensor(eye_left)
            print(f"R {R}, phi {phi}, theta {theta}", eye_right_t)
            # continue
            eye1_view = eye2world_pytroch(args.vertical, eye_left_t)
            eye2_view = eye2world_pytroch(args.vertical, eye_right_t)
            views = torch.stack([eye1_view, eye2_view])
    
            images = torch.ones(2, 3, 1080, 1920)

            images[:, :, 360:367, :] = 0
            images[:, :, 720:727, :] = 0
            images[:, :, :, 640:647] = 0
            images[:, :, :, 1280:1287] = 0


            images_ = images.cuda(non_blocking=True)[None]
            views_ = views.cuda(non_blocking=True)[None]
            with torch.no_grad():
                patterns = model.calibrate(images_, views_, FOV=40/180*math.pi).view(3, 2, 3, 1080, 1920).permute(1, 0, 2, 3, 4)


            patterns = patterns.detach().clone()
            patterns1, patterns2 = patterns[0], patterns[1]
 
            for i, pred in enumerate(patterns2):
                pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
                pred = cv2.cvtColor(np.rot90(pred, k=2), cv2.COLOR_RGB2BGR)
                pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
                cv2.imwrite(folder_path + f'view2_R{R}_phi{phi}_theta{theta}_layer-'+str(i+1)+'.png', pred)
            