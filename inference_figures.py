import os
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
from model.network import EyeRealNet
import math
import cv2
import warnings
from config.scene_dict import *
from data.dataset import eye2world_pytroch

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
        # print(f"z_min: {z_min}, type: {type(z_min)}")
        # print(f"Z_world: {Z_world}, type: {type(Z_world)}")
        # import pdb;pdb.set_trace()
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
    else:
        arg_dict = scene_dict_uco3d
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

warnings.filterwarnings('ignore')


def convert_RGB(img):
    width = img.width
    height = img.height

    image = Image.new('RGB', size=(width, height), color=(255, 255, 255))
    image.paste(img, (0, 0), mask=img)

    return image
def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
    ]

    return T.Compose(transforms)


    
def get_ori_coord(s):
    pairs = s.split('_')

    x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
    return (x,y,z)
   
def get_img_matrix(s, vertical, delta):
    ori_x, ori_y, ori_z = get_ori_coord(s)
    eye_world = torch.tensor([ori_x, ori_y, ori_z])
    return eye2world_pytroch(vertical=vertical, eye_world=eye_world, delta=delta)

def load_inference(eyeRealNet_weights, save_path, data_path, left_eye_path, right_eye_path, args, coord_screen_world):
    os.makedirs(save_path, exist_ok=True)
    FOV = args.FOV
    if FOV > math.pi: 
        FOV = FOV / 180 * math.pi
    model = EyeRealNet(args=args, FOV=FOV)
    model.load_state_dict(torch.load(eyeRealNet_weights, map_location='cpu')['model'])
    model = model.cuda()
    model.eval()
    transform = get_transform(args=args)
    # image_paths = ['view1.png', 'view2.png']
    delta = torch.tensor([args.delta_x, args.delta_y, args.delta_z])
    img_l = transform(Image.open(data_path + left_eye_path))
    img_r = transform(Image.open(data_path + right_eye_path))
    view_l = get_img_matrix(left_eye_path, args.vertical, delta)
    view_r = get_img_matrix(right_eye_path, args.vertical, delta)

    images = torch.stack([img_l, img_r], dim=0)[None]
    views = torch.stack([view_l, view_r], dim=0)[None]
    images, views = images.cuda(non_blocking=True), views.cuda(non_blocking=True)
    patterns = model(images, views, coord_screen_world)
    patterns_layer = patterns[0].detach().clone()
    for j, pred in enumerate(patterns_layer):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + 'layer-'+str(j+1)+'.png', pred)
    outs = model.get_loss(patterns, gt=images, views=views, coord_screen_world=coord_screen_world, return_preds=True) 

    preds = outs['preds'][0].detach().clone()
    for j, pred in enumerate(preds):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + 'view-'+str(j+1)+'.png', pred)


if __name__ == "__main__":
    
    from config.args import get_parser
    parser = get_parser()
    args = parser.parse_args()

    args.embed_dim = 32
    args.N_screen = 3
    args.scene = 'lego_bulldozer'
    coord_screen_world = init_scene_args(args=args)

    save_path = r"./outputs/figures/{}/".format(args.scene)
    eyeRealNet_weights = r"path/to/pretrained_model.pth"
    data_path = r'dataset/demo/lego_bulldozer/'
    left_eye_path = "pair0_x6.487_y0.237_z2.004_0.jpg"
    right_eye_path = "pair0_x6.486_y-0.263_z2.004_1.jpg"

    os.makedirs(save_path, exist_ok=True)
    load_inference(eyeRealNet_weights=eyeRealNet_weights, 
                   data_path=data_path, left_eye_path=left_eye_path, right_eye_path=right_eye_path, save_path=save_path, args=args, coord_screen_world=coord_screen_world)