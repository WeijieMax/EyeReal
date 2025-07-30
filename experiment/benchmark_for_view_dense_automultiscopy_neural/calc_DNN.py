import chainer
from chainer import Variable, serializers
import numpy as np
from PIL import Image
import os
import pickle
import time
from model import*
from ssim_DNN import *
import math
import cv2
from config.args import get_parser
from model_DNN_cupy import *
def sort_key(s):
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)
def perspective_(img, src_pts, dst_pts):
    """
    img:      ndarray, HxWxC or HxW
    src_pts:  (4, 2) float32, four points of the quadrilateral (source)
    dst_pts:  (4, 2) float32, four points after transformation (destination)
    Returns:  Perspective transformed image (same shape as img)
    """
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    matrix_H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, matrix_H, (w, h), flags=cv2.INTER_LINEAR)
    return warped
def perspective(img, src_pts, dst_pts):
	if img.ndim == 3 and img.shape[0] <= 4:  # (C, H, W)
		img = np.transpose(img, (1,2,0))     # Convert to (H,W,C) for transformation
		warped = perspective_(img, src_pts, dst_pts)
		warped = np.transpose(warped, (2,0,1))  # Convert back to (C,H,W)
	else:
		warped = perspective_(img, src_pts, dst_pts)
	return warped




def get_masks(imgs, view, coord_screen_world, coord_pixel_init):
	"""
	imgs: (N_in, H, W, C_rgb)
	view: (N_in, 4, 4)
	coord_screen_world: (N_s, 4, 3)
	coord_pixel_init:   (N_s, 4, 2)
	Returns
	masks: (N_in, H, W, C_rgb)   
	"""

	B = 1
	N_in, _, _, _ = imgs.shape

	# Repeat/broadcast imgs & view for N_s screens, flatten to first dimension
	imgs_rep = np.repeat(imgs[:, None], N_s, axis=1).reshape(N_in * N_s, H, W, C_rgb)      # (N_in*N_s, H, W, C_rgb)
	view_rep = np.repeat(view[:, None], N_s, axis=1).reshape(N_in * N_s, 4, 4)             # (N_in*N_s, 4, 4)
	# Repeat coord_screen_world, coord_pixel_init for N_in
	coord_screen_world_rep = np.tile(coord_screen_world[None, :, :, :], (N_in, 1, 1, 1)).reshape(N_in * N_s, 4, 3)
	coord_pixel_init_rep   = np.tile(coord_pixel_init[None, :, :, :],   (N_in, 1, 1, 1)).reshape(N_in * N_s, 4, 2)

	masks_new = view_transform_np(
		imgs_rep,               # (N_in*N_s, H, W, C_rgb)
		view_rep,               # (N_in*N_s, 4, 4)
		coord_screen_world_rep, # (N_in*N_s, 4, 3)
		coord_pixel_init_rep    # (N_in*N_s, 4, 2)
	)
     
	# For convenience, assume masks_new is still (N_in*N_s, H, W, C_rgb)
    # Reshape back to (N_in, N_s, H, W, C_rgb)
	masks_new = masks_new.reshape(N_in, N_s, H, W, C_rgb)
	# Perform product along N_s dimension, resulting in (N_in, H, W, C_rgb)
	masks = np.prod(masks_new, axis=1)    # (N_in, H, W, C_rgb)

	return masks

def view_transform_np(imgs, view, coord_src, coord_src_img, FOV=40/180*math.pi):

	fx = W/2 / math.tan(FOV/2)
	N = imgs.shape[0]

	# Augment coordinates to homogeneous form
	ones = np.ones((N, 4, 1), dtype=coord_src.dtype)
	coord_src_homo = np.concatenate([coord_src, ones], axis=-1)  # (N, 4, 4)

	# Calculate inverse view matrix
	view_inv = np.linalg.inv(view)  # (N, 4, 4)

	# Transform to target coordinate system
	coord_dst_homo = np.matmul(view_inv[:, None, :, :], coord_src_homo[..., None])  # (N, 4, 4, 1)
	coord_dst = np.squeeze(coord_dst_homo, -1)[..., :3]    # (N, 4, 3)

	# Project 3D coordinates to 2D pixels
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
    MSE: scalar or ndarray (can have mask)
mask: optional, 0-1 ndarray, if provided, only evaluates psnr in regions where mask is "1"
MAX: peak value, generally 1 for normalized data, 255 for uint8 images
    """
    if mask is not None:
        # Equivalent to PyTorch's mask.nelement() / mask.sum()
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
        # Note: order consistent with torch implementation
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

    # Add offset
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
        arg_dict = scene_dict_uco3d.copy()  # Avoid modifying original global dict
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
    # Generate delta using numpy array
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

C_rgb = 3
layer_num = 3
view_point_row_num = 5
view_point_num = view_point_row_num*view_point_row_num
pad_size = (int)(view_point_row_num/2)

save_path = "./outputs/experiment/DNN/calc_DNN_mask"
os.makedirs(save_path, exist_ok=True)

# Add target size definition at the beginning of the code
N_s = 3
W = 640
H = 480
target_size = (W, H)


from tqdm import tqdm
ssim_calc = SSIM()


parser = get_parser()
args = parser.parse_args()
args.scene = 'lego_bulldozer'
coord_screen_world = init_scene_args_numpy(args=args)
delta = np.array([args.delta_x, args.delta_y, args.delta_z])
coord_pixel_init = np.array([(0, 0), (W, 0), (0, H), (W, H)], dtype=np.float32)
coord_pixel_init = np.tile(coord_pixel_init[None, :, :], (3, 1, 1))
for i in range(0, 7):
	
	data_path = r"dataset\experiment\DNN\lego_bulldozer\lego_bulldozer200_25_scale_0.083_R_{}_{}_FOV_40_theta_40_140_phi_60_120".format(10+20*i, 30+20*i)


	image_names = os.listdir(data_path)
	image_names = sorted(image_names, key=sort_key)
	img_num = len(image_names) // 25


	psnr_ls = list()
	ssim_ls = list()
	time_ls = list()
	for j in tqdm(range(img_num)):
		


		#Load light field datas
		cpu_light_field = []
		cpu_layer = []
		cpu_reproduced_light_field = []
		views = []

		for t in range(view_point_num):
			img_name = image_names[j*25 + t]
			fname = os.path.join(data_path, img_name)
			views.append(get_img_matrix_numpy(img_name, args.vertical, delta))
			img_load = Image.open(fname).resize(target_size, Image.LANCZOS)
			img_load = np.asarray(img_load)
			img_load = img_load.astype(np.float32)/255.0
			cpu_light_field.append(img_load.copy())
		views = np.stack(views)
		masks = get_masks(np.stack(cpu_light_field), views, coord_screen_world, coord_pixel_init)
		size_h = cpu_light_field[0].shape[0]
		size_w = cpu_light_field[0].shape[1]

		for t in range(layer_num):
			cpu_layer.append(np.zeros((size_h, size_w, C_rgb)))

		for t in range(view_point_num):
			cpu_reproduced_light_field.append(np.zeros((size_h-pad_size*2, size_w-pad_size*2, C_rgb)))



		#Model initaition
		calclayer = CalcLayer()
		serializers.load_npz(r'experiment\DNN\weights\calclayer_origin_paper.npz', calclayer)
		calclayer.to_gpu()


		#Calculate layer
		gpu_light_field = np.zeros((C_rgb, view_point_num, size_h, size_w), np.float32)

		for ch in range(C_rgb):
			for t in range(view_point_num):
				gpu_light_field[ch, t, :, :] = cpu_light_field[t][:, :, ch]

		gpu_light_field = chainer.cuda.to_gpu(gpu_light_field)
		gpu_light_field = Variable(gpu_light_field)

		st_time = time.time()
		gpu_layer = calclayer(gpu_light_field)
		# end_time = time.time()
		# time_ls.append(end_time - st_time)

		gpu_layer=F.expand_dims(gpu_layer, 2)
		
		gpu_reproduced_light_field = gpu_layer[:, 0, :, 0:size_h-pad_size*2, 0:size_w-pad_size*2]
		end_time = time.time()
		time_ls.append(end_time - st_time)
		print(end_time - st_time)
		for p in range (view_point_row_num):
			for t in range (view_point_row_num):
				gpu_reproduced_light_field = F.concat([gpu_reproduced_light_field, gpu_layer[:, 0, :, p:size_h-pad_size*2+p, t:size_w-pad_size*2+t] + gpu_layer[:, 1, :, pad_size:size_h-pad_size, pad_size:size_w-pad_size] + gpu_layer[:, 2, :, pad_size*2-p:size_h-p, pad_size*2-t:size_w-t]], axis=1)
		
		gpu_reproduced_light_field = gpu_reproduced_light_field.data
		gpu_layer = gpu_layer.data

		gpu_layer = chainer.cuda.to_cpu(gpu_layer)
		gpu_reproduced_light_field = chainer.cuda.to_cpu(gpu_reproduced_light_field)


		for ch in range(C_rgb):
			for t in range(layer_num):
				cpu_layer[t][:,:,ch] = gpu_layer[ch,t,:,:]
			for u in range(view_point_num):
				cpu_reproduced_light_field[u][:,:,ch] = gpu_reproduced_light_field[ch,u+1,:,:]


		#Save reproduced light field
		
		import cupy as cp
		gpu_reproduced = cp.asarray(np.stack(cpu_reproduced_light_field[:view_point_num]))
		gpu_gt = cp.asarray(np.stack([img[2:size_h-2, 2:size_w-2, :] 
                              for img in cpu_light_field[:view_point_num]]))                                                      
		gpu_reproduced = cp.clip(gpu_reproduced, 0, 1)

		masks_ = cp.asarray(masks[:, 2:size_h-2, 2:size_w-2, :])

		diff = gpu_gt * masks_ - gpu_reproduced * masks_
		mse_per_image = cp.mean(diff**2, axis=(1, 2, 3))
		psnr_per_image = get_PSNR(mse_per_image, masks_)

		# SSIM calculation (batch processing)
		ssim_per_image = ssim_calc(gpu_reproduced*255*masks_, gpu_gt*255*masks_)

		# 5. Calculate average metrics
		avg_psnr = cp.mean(psnr_per_image).get()
		avg_ssim = cp.mean(ssim_per_image).get()
		psnr_ls.append(avg_psnr)
		ssim_ls.append(avg_ssim)

	psnr_arr = np.array(psnr_ls)
	ssim_arr = np.array(ssim_ls)
	time_arr = np.array(time_ls)
	exp_name = 'R_{}_{}'.format(10+20*i, 30+20*i)

	output_path = os.path.join(save_path, exp_name)
	os.makedirs(output_path, exist_ok=True)

	# Save as NumPy .npy format
	np.save(os.path.join(output_path, "{}_psnr.npy".format(exp_name)), psnr_arr)
	np.save(os.path.join(output_path, "{}_ssim.npy".format(exp_name)), ssim_arr)
	np.save(os.path.join(output_path, "{}_time.npy".format(exp_name)), time_arr)

	# Write text file with statistical results
	with open(os.path.join(output_path, 'output.txt'), 'a') as f:
		f.write("{}_all_psnr: {:.6f}\n".format(exp_name, np.sum(psnr_arr)))
		f.write("{}_all_ssim: {:.6f}\n".format(exp_name, np.sum(ssim_arr)))
		f.write("{}_all_time: {:.6f}\n".format(exp_name, np.sum(time_arr)))
		f.write("{}_avg_psnr: {:.6f}\n".format(exp_name, np.mean(psnr_arr)))
		f.write("{}_avg_ssim: {:.6f}\n".format(exp_name, np.mean(ssim_arr)))
		f.write("{}_avg_time: {:.6f}\n".format(exp_name, np.mean(time_arr)))
		f.write("{}_stddev_psnr: {:.6f}\n".format(exp_name, np.std(psnr_arr)))
		f.write("{}_stddev_ssim: {:.6f}\n".format(exp_name, np.std(ssim_arr)))
		f.write("{}_stddev_time: {:.6f}\n".format(exp_name, np.std(time_arr)))



