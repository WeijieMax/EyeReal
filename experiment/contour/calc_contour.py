from config.scene_dict import scene_dict
from config.args import get_parser, get_gaussian_parser
from model.network import EyeRealNet
import math
import torch
from data.render import GaussianRender
from torchvision.transforms.functional import perspective
import cv2
import copy
from torch.nn import functional as F
from model.metric import get_PSNR
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
import numpy as np
def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
    ]

    return T.Compose(transforms)
def eye2world_pytroch(vertical, eye_world: torch.Tensor, delta=torch.tensor([0., 0., 0.])):
    eye_world = eye_world.float()
    vecz = eye_world
    vecz = vecz / torch.linalg.norm(vecz)
    if vertical == "z":
        vec_w = torch.tensor([1e-5, 1e-6, 1.]).to(eye_world.device)
    elif vertical == "x":
        vec_w = torch.tensor([1., 1e-6, 1e-5]).to(eye_world.device)
    elif vertical == "-x":
        vec_w = torch.tensor([-1., 1e-6, 1e-5]).to(eye_world.device)
    elif vertical == "y":
        vec_w = torch.tensor([1e-6, 1., 1e-5]).to(eye_world.device)
    else:
        raise ValueError("wrong input vertical")

    vecx = torch.cross(vec_w, vecz)
    vecx = vecx / torch.linalg.norm(vecx)
    vecy = torch.cross(vecz, vecx)
    vecy = vecy / torch.linalg.norm(vecy)
    rot = torch.stack([vecx, vecy, vecz]).T
    rt = torch.eye(4).to(eye_world.device)
    rt[:3, :3] = rot
    rt[:3, 3] = eye_world + delta
    
    return rt
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
def load_model(args):
    FOV = args.FOV
    if FOV > math.pi:
        FOV = FOV / 180 * math.pi
    model = EyeRealNet(args=args, FOV=FOV)
    model = model.cuda()
    if args.ckpt_weights:
        checkpoint = torch.load(args.ckpt_weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.eval()
    return model
def left_coords_dict(step=2, boundary=6, eye=None, args=None):
    
    c_d = {}
    views_d = {}

    c_d[0] = [torch.tensor(eye)]
    vertical = args.vertical
    delta = torch.tensor([args.delta_x, args.delta_y, args.delta_z])

    for r in range(step, boundary+step, step):
        for theta in range(0, 360, 45):
            theta_ = math.radians(theta)
            x = eye[0]
            y = eye[1] + r * math.cos(theta_)
            z = eye[2] + r * math.sin(theta_)
            if y < 0:
                if c_d.get(r) != None:
                    c_d[r].append(torch.tensor([x,y,z]))
                else:
                    c_d[r] = [torch.tensor([x,y,z])]
    for r in range(0, boundary+step, step):
        c_d[r] = torch.stack(c_d[r])
        c_d[r] = c_d[r] * args.scale_physical2world

    for r in range(0, boundary+step, step):
        views_d[r] = []
        for coords in c_d[r]:
            # import pdb;pdb.set_trace()
            views_d[r].append(eye2world_pytroch(vertical=vertical, eye_world=coords, delta=delta))
        
    return c_d, views_d
def right_coords_dict(step=2, boundary=6, eye=None, args=None):
    c_d = {}
    views_d = {}
    c_d[0] = [torch.tensor(eye)]

    vertical = args.vertical
    delta = torch.tensor([args.delta_x, args.delta_y, args.delta_z])

    for r in range(step, boundary+step):
        for theta in range(0, 360, 45):
            theta_ = math.radians(theta)
            x = eye[0]
            y = eye[1] + r * math.cos(theta_)
            z = eye[2] + r * math.sin(theta_)
            if y > 0:
                if c_d.get(r) != None:
                    c_d[r].append(torch.tensor([x,y,z]))
                else:
                    c_d[r] = [torch.tensor([x,y,z])]
    for r in range(0, boundary+step, step):
        c_d[r] = torch.stack(c_d[r])
        c_d[r] = c_d[r] * args.scale_physical2world
    for r in range(0, boundary+step, step):
        views_d[r] = []
        for coords in c_d[r]:
            views_d[r].append(eye2world_pytroch(vertical=vertical, eye_world=coords, delta=delta))
    return c_d, views_d
def get_gaussian_render(FOV, gaussians_path):
    render = GaussianRender(
        parser=get_gaussian_parser(),
        sh_degree=3, 
        gaussians_path=gaussians_path,
        white_background=True, FOV=FOV / 180 * math.pi)
    return render

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
def perspective_cam2cam(img, view1, view2, mask, FOV, scale_pixel2world, coords_init):
    # view1 : from view
    # view2 : to view
    _, H, W = img.shape
    fx = W/2 / math.tan(FOV/2)


    img_coords1p = torch.tensor([
        [-W/2, H/2, -fx],
        [W/2, H/2, -fx],
        [-W/2, -H/2, -fx],
        [W/2, -H/2, -fx],
    ])
    img_coords1c = img_coords1p * scale_pixel2world

    img_coords1c_homo = torch.cat([img_coords1c, torch.ones(4,1)], dim=-1).to(img.device)

    ###
    view1 = view1.to(img.device)
    view2 = view2.to(img.device)
    ###

    img_coords1w_homo = torch.matmul(view1[None], img_coords1c_homo[..., None]).squeeze(-1) # N 4 3
    img_coords2c_homo = torch.matmul(torch.inverse(view2)[None], img_coords1w_homo[..., None]).squeeze(-1)[..., :3] # N 4 3
    u = (-fx*img_coords2c_homo[..., [0]]/img_coords2c_homo[..., [2]] + W/2)
    v = (fx*img_coords2c_homo[..., [1]]/img_coords2c_homo[..., [2]] + H/2)
    img_coords2p = torch.cat([u, v], dim=-1)


    
    img_new = perspective(img, img_coords2p.tolist(), coords_init.tolist()) 
    mask_new = perspective(mask, img_coords2p.tolist(), coords_init.tolist()) 

    return img_new, mask_new

def calc_view2view_psnr(boundary, step, views_dict, img, mask, render, args, coords_init, scale_pixel2world):
    psnr_dict = {}
    src_view = views_dict[0][0].cuda()


    for r in range(0, boundary+step, step):
        psnr_dict[r] = []
        if r == 0:
            src_img_gt = render.render_from_view(copy.deepcopy(src_view))

            mse = F.mse_loss(img*mask, src_img_gt*mask)
            psnr = get_PSNR(mse.item(), mask)
            psnr_dict[r].append(psnr)
            
        else:
            for dst_view in views_dict[r]:
                dst_view = dst_view.cuda()
                dst_img, dst_mask = perspective_cam2cam(img=copy.deepcopy(img), view1=src_view, view2=dst_view, mask=copy.deepcopy(mask), 
                                    FOV=args.FOV/180*math.pi, scale_pixel2world=scale_pixel2world, coords_init=coords_init)
                dst_img_gt = render.render_from_view(copy.deepcopy(dst_view))
                mse = F.mse_loss(dst_img*dst_mask, dst_img_gt*dst_mask)
                psnr = get_PSNR(mse.item(), dst_mask)
                psnr_dict[r].append(psnr)
    return psnr_dict

def get_img(view):
    img = render.render_from_view(copy.deepcopy(view))
    save_image(img, 'test.jpg')
    img_ = transform(Image.open('test.jpg'))
    return img_

def get_contour_pts():

    
    
    

    x, z = view_1_coords[0], view_1_coords[2] - edge_b
    y = view_1_coords[1] - edge_a
    
    pts = []
    views = []
    
    for i in range(b_num + 1):
        st_pt = [x, y, z+step*i]
        i_pts = []
        i_views = []
        for j in range(a_num + 1):
            coords = torch.tensor([st_pt[0], st_pt[1]+step*j, st_pt[2]])
            coords_ = coords * scale_physical2world
            i_pts.append(coords * 10)
            i_views.append(eye2world_pytroch(vertical=vertical, eye_world=coords_, delta=delta))
        i_pts = torch.stack(i_pts)
        i_views = torch.stack(i_views)
        pts.append(i_pts)
        views.append(i_views)
    pts = torch.stack(pts)
    views = torch.stack(views)
    return pts, views

def get_model_pred(left_view, right_view):
    left_view = left_view.cuda()
    right_view = right_view.cuda()

    left_img = get_img(left_view)
    right_img = get_img(right_view)
    
    images = torch.stack([left_img, right_img], dim=0)[None]
    views = torch.stack([left_view, right_view], dim=0)[None]
    # images = torch.stack([right_img,left_img], dim=0)[None]
    # views = torch.stack([right_view,left_view], dim=0)[None]
    images, views = images.cuda(non_blocking=True), views.cuda(non_blocking=True)
    coord_screen_world_ = coord_screen_world[None].cuda()
    # import pdb;pdb.set_trace()
    patterns = model(images, views, coord_screen_world_)
    outs = model.get_loss(patterns, gt=images, views=views, coord_screen_world=coord_screen_world, return_preds=True) 
    
    return patterns.detach().clone(), outs['preds'][0].detach().clone(), outs['masks'][0].detach().clone()

def get_pattern_contour(pts, views, coord_screen_world_):
    
    patterns, _, _ = get_model_pred(eye2world_pytroch(vertical=vertical, eye_world=torch.tensor(view_0_coords)*scale_physical2world, delta=delta), eye2world_pytroch(vertical=vertical, eye_world=torch.tensor(view_1_coords)*scale_physical2world, delta=delta))

    psnrs = []
    for i_views in views:
        i_psnrs = []
        for view in i_views:
            view = view.cuda()
            img_gt = render.render_from_view(copy.deepcopy(view)).cuda()
            outs = model.get_loss(patterns, gt=img_gt[None], views=view[None][None], coord_screen_world=coord_screen_world_, return_preds=False) 
            i_psnrs.append(outs['PSNR'])
        psnrs.append(i_psnrs)
    
    xi = []
    yi = []
    zi = []
    for i in range(b_num + 1):
        xi.append([])
        yi.append([])
        zi.append([])
        for j in range(a_num + 1):
            xi[i].append(pts[i][j][1])
            yi[i].append(pts[i][j][2])
            zi[i].append(psnrs[i][j])
    print("contour_pattern: ")
    print('xi: ', xi)
    print('yi: ', i)
    print('zi: ', zi)


def get_side_view(views, img, mask, src_view):
    psnrs = []
    for i_views in views:
        i_psnrs = []
        for dst_view in i_views:
            dst_view = dst_view.cuda()
            dst_img, dst_mask = perspective_cam2cam(img=copy.deepcopy(img), view1=src_view, view2=dst_view, mask=copy.deepcopy(mask), 
                                    FOV=args.FOV/180*math.pi, scale_pixel2world=scale_pixel2world, coords_init=coords_init)
            dst_img_gt = render.render_from_view(copy.deepcopy(dst_view))
            mse = F.mse_loss(dst_img*dst_mask, dst_img_gt*dst_mask)
            psnr = get_PSNR(mse.item(), dst_mask)
            i_psnrs.append(psnr)
        psnrs.append(i_psnrs)
    return psnrs

def get_view_contour(pts, views, coord_screen_world_):

    _, key_imgs_, key_mask_ = get_model_pred(eye2world_pytroch(vertical=vertical, eye_world=torch.tensor(view_0_coords)*scale_physical2world, delta=delta), eye2world_pytroch(vertical=vertical, eye_world=torch.tensor(view_1_coords)*scale_physical2world, delta=delta))
    key_imgs = key_imgs_.flip(0)
    key_masks = key_mask_.flip(0)

    left_psnrs = get_side_view(views=views[:, 0:a_num//2], img=key_imgs[0], mask=key_masks[0], src_view=views[edge_b_num][edge_a_num])
    right_psnrs = get_side_view(views=views[:, a_num//2:], img=key_imgs[1], mask=key_masks[1], src_view=views[edge_b_num][edge_a_num + mid_num])

    psnrs = [left_psnrs[i]+right_psnrs[i] for i in range(len(left_psnrs))]
    
    xi = []
    yi = []
    zi = []
    for i in range(b_num + 1):
        xi.append([])
        yi.append([])
        zi.append([])
        for j in range(a_num + 1):
            xi[i].append(pts[i][j][1])
            yi[i].append(pts[i][j][2])
            zi[i].append(psnrs[i][j])

    refine_dis = 2
    left_psnrs_refine = get_side_view(views=views[:, a_num//2 - refine_dis : a_num//2 + refine_dis + 1], img=key_imgs[0], mask=key_masks[0], src_view=views[edge_b_num][edge_a_num])
    right_psnrs_refine = get_side_view(views=views[:, a_num//2 - refine_dis : a_num//2 + refine_dis + 1], img=key_imgs[1], mask=key_masks[1], src_view=views[edge_b_num][edge_a_num + mid_num])

    left_psnrs_refine_np = np.array(left_psnrs_refine)
    right_psnrs_refine_np = np.array(right_psnrs_refine)

    refine_np = (left_psnrs_refine_np + right_psnrs_refine_np) / 2
    zi=np.array(zi)
    zi[:, a_num//2 - refine_dis : a_num//2 + refine_dis + 1] = refine_np

    print("contour_perspective: ")
    print('xi: ', xi)
    print('yi: ', i)
    print('zi: ', zi)


def get_contour():
    pts, views = get_contour_pts()
    get_pattern_contour(pts=pts, views=views, coord_screen_world_=coord_screen_world[None].cuda())
    get_view_contour(pts=pts, views=views, coord_screen_world_=coord_screen_world[None].cuda())


def get_img_matrix(s):
    ori_x, ori_y, ori_z = get_ori_coord(s)
    eye_world = torch.tensor([ori_x, ori_y, ori_z])
    return eye2world_pytroch(vertical='z', eye_world=eye_world)

def get_ori_coord(s):
    # "pair0_x5.896_y45.396_z-24.465_left"
    pairs = s.split('_')

    x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
    return (x,y,z)


# 定义阈值和边界
LOW_THRESH = 20         # 低阈值
HIGH_THRESH = 34        # 高阈值
mid = 6
edge_b = 6
edge_a = 5
a = mid + edge_a * 2
b = edge_b * 2
step = 0.02

mid_num = int(mid / step)
edge_b_num = int(edge_b / step)
edge_a_num = int(edge_a / step)
a_num = int(a / step)
b_num = int(b / step)

parser = get_parser()
args = parser.parse_args()
args.scene = 'lego_bulldozer'
args.FOV = 40
args.embed_dim = 32
args.model_choice = 0

init_scene_args(args=args)

args.ckpt_weights = r"weight\model_ckpts\pretrained_model.pth"
gaussians_path = 'weight\gaussian_ply\lego_bulldozer.ply'
model = load_model(args)
render = get_gaussian_render(args.FOV, gaussians_path)
transform = get_transform(args=args)

view_1_coords = [75, -3, 0]
view_0_coords = [75, 3, 0]

vertical = args.vertical
delta = torch.tensor([args.delta_x, args.delta_y, args.delta_z])
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
scale_physical2world = args.scale_physical2world
scale_pixel2world = scale_physical2world * args.physical_width / args.image_width
coords_init=model.coord_pixel_init[0]




get_contour()
