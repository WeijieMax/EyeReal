import torch
from model_DNN_tensor import CalcLayer as CalcLayerTorch
import numpy as np
from PIL import Image
import os
import pickle
import time
# from metric_gpu import *
import math
import cv2
from config.args import get_parser
import gc
from data.render import GaussianRender
from config.args import get_parser,get_gaussian_parser
def eye2world_pytroch(eye_world: torch.Tensor):
    eye_world = eye_world.float()
    vecz = eye_world
    vecz = vecz / torch.linalg.norm(vecz)

    vec_w = torch.tensor([1e-5, 1e-6, 1.]).to(eye_world.device) 

    vecx = torch.cross(vec_w, vecz)
    vecx = vecx / torch.linalg.norm(vecx)
    vecy = torch.cross(vecz, vecx)
    vecy = vecy / torch.linalg.norm(vecy)
    rot = torch.stack([vecx, vecy, vecz]).T
    rt = torch.eye(4).to(eye_world.device)
    rt[:3, :3] = rot
    rt[:3, 3] = eye_world + delta
    
    return rt
def sort_key(s):
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)
def perspective_(img, src_pts, dst_pts):
    """
    img:      ndarray, HxWxC or HxW
    src_pts:  (4, 2) float32, 四边形的四个点 (源)
    dst_pts:  (4, 2) float32, 四边形变换后的四个点 (目标)
    返回:      透视变换后的图像 (与 img 相同 shape)
    """
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    matrix_H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, matrix_H, (w, h), flags=cv2.INTER_LINEAR)
    return warped
def perspective(img, src_pts, dst_pts):
	if img.ndim == 3 and img.shape[0] <= 4:  # (C, H, W)
		img = np.transpose(img, (1,2,0))     # 转成 (H,W,C) 再变换
		warped = perspective_(img, src_pts, dst_pts)
		warped = np.transpose(warped, (2,0,1))  # 转回 (C,H,W)
	else:
		warped = perspective_(img, src_pts, dst_pts)
	return warped




def get_masks(imgs, view, coord_screen_world, coord_pixel_init):
	"""
	imgs: (N_in, H, W, C_rgb)
	view: (N_in, 4, 4)
	coord_screen_world: (N_s, 4, 3)
	coord_pixel_init:   (N_s, 4, 2)
	返回
	masks: (N_in, H, W, C_rgb)   
	"""

	B = 1
	N_in, _, _, _ = imgs.shape

	# Repeat/broadcast imgs & view for N_s screens, flatten为第0维
	imgs_rep = np.repeat(imgs[:, None], N_s, axis=1).reshape(N_in * N_s, H, W, C_rgb)      # (N_in*N_s, H, W, C_rgb)
	view_rep = np.repeat(view[:, None], N_s, axis=1).reshape(N_in * N_s, 4, 4)             # (N_in*N_s, 4, 4)
	# coord_screen_world, coord_pixel_init 对 N_in 重复
	coord_screen_world_rep = np.tile(coord_screen_world[None, :, :, :], (N_in, 1, 1, 1)).reshape(N_in * N_s, 4, 3)
	coord_pixel_init_rep   = np.tile(coord_pixel_init[None, :, :, :],   (N_in, 1, 1, 1)).reshape(N_in * N_s, 4, 2)

	masks_new = view_transform_np(
		imgs_rep,               # (N_in*N_s, H, W, C_rgb)
		view_rep,               # (N_in*N_s, 4, 4)
		coord_screen_world_rep, # (N_in*N_s, 4, 3)
		coord_pixel_init_rep    # (N_in*N_s, 4, 2)
	)
     
	# 为方便，假设 masks_new 还是 (N_in*N_s, H, W, C_rgb)
    # reshape 回合成 (N_in, N_s, H, W, C_rgb)
	masks_new = masks_new.reshape(N_in, N_s, H, W, C_rgb)
	# 在 N_s 维度上做乘积（prod），得到 (N_in, H, W, C_rgb)
	masks = np.prod(masks_new, axis=1)    # (N_in, H, W, C_rgb)

	return masks

def view_transform_np(imgs, view, coord_src, coord_src_img, FOV=40/180*math.pi):

	fx = W/2 / math.tan(FOV/2)
	N = imgs.shape[0]

	# 增广坐标为齐次形式
	ones = np.ones((N, 4, 1), dtype=coord_src.dtype)
	coord_src_homo = np.concatenate([coord_src, ones], axis=-1)  # (N, 4, 4)

	# 计算逆view矩阵
	view_inv = np.linalg.inv(view)  # (N, 4, 4)

	# 变换到目标坐标系
	coord_dst_homo = np.matmul(view_inv[:, None, :, :], coord_src_homo[..., None])  # (N, 4, 4, 1)
	coord_dst = np.squeeze(coord_dst_homo, -1)[..., :3]    # (N, 4, 3)

	# 将3D坐标投影到2D像素
	u = (-fx * coord_dst[..., [0]] / coord_dst[..., [2]] + W/2)  # (N, 4, 1)
	v = (fx * coord_dst[..., [1]] / coord_dst[..., [2]] + H/2)   # (N, 4, 1)
	coord_dst_img = np.concatenate([u, v], axis=-1)   # (N, 4, 2)


	masks = np.ones_like(imgs)

	masks_new = np.stack([
		perspective(mask, src, dst)
		for mask, src, dst in zip(masks, coord_src_img, coord_dst_img)
	])
	return masks_new




def eye2world_numpy(vertical, eye_world: np.ndarray, delta=np.array([0., 0., 0.])):
    eye_world = eye_world.astype(np.float32)
    vecz = eye_world / np.linalg.norm(eye_world)
    if vertical == "z":
        vec_w = np.array([1e-5, 1e-6, 1.], dtype=np.float32)
    elif vertical == "x":
        vec_w = np.array([1., 1e-6, 1e-5], dtype=np.float32)
    elif vertical == "-x":
        vec_w = np.array([-1., 1e-6, 1e-5], dtype=np.float32)
    elif vertical == "y":
        vec_w = np.array([1e-6, 1., 1e-5], dtype=np.float32)
    else:
        raise ValueError("wrong input vertical")
    vecx = np.cross(vec_w, vecz)
    vecx = vecx / np.linalg.norm(vecx)
    vecy = np.cross(vecz, vecx)
    vecy = vecy / np.linalg.norm(vecy)
    rot = np.stack([vecx, vecy, vecz], axis=1)    # shape (3,3)
    rt = np.eye(4, dtype=np.float32)
    rt[:3, :3] = rot
    rt[:3, 3] = eye_world + delta
    return rt
def get_PSNR(MSE, mask=None, MAX=1):
    """
    MSE: 标量，或 ndarray（可带 mask）
    mask: 可选，0-1 ndarray，若给出则仅对 mask “为 1”的区域做 psnr 评估
    MAX: 峰值，一般归一化情况下为 1，uint8 图片时为 255
    """
    if mask is not None:
        # 等价于 PyTorch 的 mask.nelement() / mask.sum()
        MSE = MSE * mask.size / mask.sum()
    return 10 * np.log10(MAX ** 2 / MSE)
def get_screen_coords_world_numpy(thickness, scale_physical2world, physical_width, ground, ground_coefficient, orientation, delta):


    

    if physical_width is None:
        physical_width = 51.84
    if ground is None:
        ground = 0

    scale_pixel2world = scale_physical2world * physical_width / W
    Z_world = thickness * scale_physical2world

    if ground_coefficient is not None:
        z_min = Z_world * ground_coefficient
    else:
        z_min = ground
    z_max = z_min + Z_world
    W_w = W * scale_pixel2world
    H_w = H * scale_pixel2world

    zs = np.linspace(z_min, z_max, N_s)
    coords = []
    if orientation == "xoy":
        # 注意顺序与torch实现一致
        for z in zs:
            coords.append([
                [-W_w/2, H_w/2, z],
                [ W_w/2, H_w/2, z],
                [-W_w/2,-H_w/2, z],
                [ W_w/2,-H_w/2, z]
            ])
    elif orientation == "xoz":
        for z in zs:
            coords.append([
                [ -W_w/2, z, H_w/2],
                [  W_w/2, z, H_w/2],
                [ -W_w/2, z,-H_w/2],
                [  W_w/2, z,-H_w/2],
            ])
    elif orientation == "yox":
        for z in zs:
            coords.append([
                [-H_w/2, -W_w/2, z],
                [-H_w/2,  W_w/2, z],
                [ H_w/2, -W_w/2, z],
                [ H_w/2,  W_w/2, z]
            ])
    elif orientation == "yoz":
        for z in zs:
            coords.append([
                [z, -W_w/2,  H_w/2],
                [z,  W_w/2,  H_w/2],
                [z, -W_w/2, -H_w/2],
                [z,  W_w/2, -H_w/2],
            ])
    elif orientation == "zox":
        for z in zs:
            coords.append([
                [ H_w/2, z, -W_w/2],
                [ H_w/2, z,  W_w/2],
                [-H_w/2, z, -W_w/2],
                [-H_w/2, z,  W_w/2],
            ])
    else:
        raise ValueError("Unsupported orientation.")

    coord_screen_world = np.array(coords, dtype=np.float32) # (N_screen, 4, 3)

    # 加偏移
    delta = np.array(delta)
    for index in range(3):
        coord_screen_world[..., index] += delta[index]
    return coord_screen_world

def get_ori_coord(s):
    pairs = s.split('_')
    x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
    return (x,y,z)
def get_img_matrix_numpy(s, vertical, delta):
    ori_x, ori_y, ori_z = get_ori_coord(s)
    eye_world = np.array([ori_x, ori_y, ori_z], dtype=np.float32)
    return eye2world_numpy(vertical=vertical, eye_world=eye_world, delta=np.array(delta, dtype=np.float32))

from config.scene_dict import *
def init_scene_args_numpy(args):
    if args.scene in scene_dict:
        arg_dict = scene_dict[args.scene]
    else:
        arg_dict = scene_dict_uco3d.copy()  # 避免修改原始全局dict
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
    # 用 numpy array 生成 delta
    delta = np.zeros(3, dtype=np.float32)
    delta[0] = arg_dict.get('delta_x', 0)
    delta[1] = arg_dict.get('delta_y', 0)
    delta[2] = arg_dict.get('delta_z', 0)

    coord_screen_world = get_screen_coords_world_numpy(
        thickness = arg_dict.get('thickness'),
        scale_physical2world = arg_dict.get('scale_physical2world'),
        physical_width = arg_dict.get('physical_width'),
        ground = arg_dict.get('ground'),
        ground_coefficient = arg_dict.get('ground_coefficient'),
        orientation = arg_dict.get('orientation'),
        delta = delta
    )
    return coord_screen_world

def get_25_from_2(eye1, eye2):

    # 2. 计算线段eye1-eye2的四等分点
    AB = eye2 - eye1
    d = torch.norm(AB) / 2  # 四分之一距离

    # 计算线段上的5个点 (使用PyTorch张量运算)
    points_on_line = [
        eye1 - AB * 0.5,
        eye1,
        eye1 + AB * 0.5,
        eye2,
        eye2 + AB * 0.5,
    ]

    # 3. 计算垂直方向
    A = AB  # 向量A (eye1->eye2)
    B = points_on_line[2]  # 向量B (指向中点q2)
    C = torch.cross(A, B)  # 向量C = A × B (使用PyTorch叉积)

    # 归一化处理
    norm_C = torch.norm(C)
    if norm_C < 1e-10:  # 处理叉积为零的情况
        alt_vector = torch.tensor([1.0, 0.0, 0.0] if abs(A[0]) < 0.9 else [0.0, 1.0, 0.0])
        C = torch.cross(A, alt_vector)
        norm_C = torch.norm(C)
        if norm_C < 1e-10:
            C = torch.tensor([0.0, 0.0, 1.0])

    unit_C = C / norm_C

    # 4. 生成25个点并创建文件名
    # k_values = [-2, -1, 0, 1, 2]
    k_values = [2, 1, 0, -1, -2]
    pts_ls = []
    for i, k in enumerate(k_values):
        for j, point in enumerate(points_on_line):
            offset_point = point + k * d * unit_C
            pts_ls.append(offset_point)
    return pts_ls

C_rgb = 3
layer_num = 3
view_point_row_num = 5
view_point_num = view_point_row_num*view_point_row_num
pad_size = (int)(view_point_row_num/2)



# 在代码开头添加目标尺寸定义
N_s = 3
W = 640
H = 480
target_size = (W, H)



render = GaussianRender(
    parser=get_gaussian_parser(),
    sh_degree=3, 
    gaussians_path=r"weight\gaussian_ply\lego_bulldozer.ply",
    white_background=True, FOV=40 / 180 * math.pi, render_image_size=(H,W))


from tqdm import tqdm
# ssim_calc = SSIM()


parser = get_parser()
args = parser.parse_args()
args.scene = 'lego_bulldozer'
coord_screen_world = init_scene_args_numpy(args=args)
delta = np.array([args.delta_x, args.delta_y, args.delta_z])
coord_pixel_init = np.array([(0, 0), (W, 0), (0, H), (W, H)], dtype=np.float32)
coord_pixel_init = np.tile(coord_pixel_init[None, :, :], (3, 1, 1))



data_path = 'dataset\eval\lego_bulldozer\lego_bulldozer200_scale_0.083_R_10_30_FOV_40_theta_40_140_phi_60_120/'
left_img_name = 'pair0_x0.944_y-0.737_z-0.634_0.jpg'
right_img_name = 'pair0_x0.563_y-1.057_z-0.634_1.jpg' 


psnr_ls = list()
# ssim_ls = list()
time_ls = list()

eye1 = get_ori_coord(left_img_name)
eye2 = get_ori_coord(right_img_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    for i in tqdm(range(1000)):

        
        #######################################
        
        st_time = time.time()
        coord_ls = get_25_from_2(eye1=torch.tensor(eye1), eye2=torch.tensor(eye2))
        # step1: 采集25个视角图片, 假设render.render_from_view输出(C, H, W), 不修改
        tensor_imgs = []
        for i in range(25):
            tensor_view = eye2world_pytroch(coord_ls[i])             
            img = render.render_from_view(tensor_view.cuda())         
            tensor_imgs.append(img)
        tensor_imgs = torch.stack(tensor_imgs)         # (25, C, H, W)


        tensor_imgs = tensor_imgs.permute(1, 0, 2, 3)  # (C, 25, H, W)

        calclayer = CalcLayerTorch().to(device)
        calclayer.eval()
        with torch.no_grad():
            x = tensor_imgs.cuda()               
            gpu_layer = calclayer(x)              
        end_time = time.time()
        time_ls.append(end_time - st_time)
        print(end_time - st_time)

        del calclayer, gpu_layer
        torch.cuda.empty_cache()
print('cnn tensor time mean is {}, FPS is {}'.format(torch.tensor(time_ls).mean(), 1 / torch.tensor(time_ls).mean()))