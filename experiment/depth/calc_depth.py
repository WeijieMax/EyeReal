
from PIL import Image
import torch
from torchvision import transforms as T
from torch.nn import functional as F
from model.metric import get_PSNR
import torch, math
from torchvision.transforms.functional import perspective
import os
def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
    ]

    return T.Compose(transforms)

def sort_key(s):
    # "pair0_x5.896_y45.396_z-24.465_left.jpg"
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)


def view_tranform(imgs, view, coord_src, coord_src_img, reverse=False):
    N, _, H, W = imgs.shape
    coord_src_homo = torch.cat([coord_src, torch.ones(N,4,1)], dim=-1).to(imgs.device)
    coord_dst = torch.matmul(torch.inverse(view)[:, None], coord_src_homo[..., None]).squeeze(-1)[..., :3] # N 4 3
    u = (-fx*coord_dst[..., [0]]/coord_dst[..., [2]] + W/2)
    v = (fx*coord_dst[..., [1]]/coord_dst[..., [2]] + H/2)
    coord_dst_img = torch.cat([u, v], dim=-1)

    masks = torch.ones_like(imgs)
    if not reverse:
        imgs_new = torch.stack([perspective(img, src.tolist(), dst.tolist()) 
                            for img, src, dst in zip(imgs, coord_src_img, coord_dst_img)])
        masks_new = torch.stack([perspective(mask, src.tolist(), dst.tolist()) 
                            for mask, src, dst in zip(masks, coord_src_img, coord_dst_img)])
    else:
        imgs_new = torch.stack([perspective(img, src.tolist(), dst.tolist()) 
                            for img, src, dst in zip(imgs, coord_dst_img, coord_src_img)])
        masks_new = torch.stack([perspective(mask, src.tolist(), dst.tolist()) 
                            for mask, src, dst in zip(masks, coord_dst_img, coord_src_img)])

    return imgs_new, masks_new
def eye2world_pytroch(eye_world: torch.Tensor):

    vecz = eye_world.float()
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
def get_ori_coord(s):
    # "pair0_x5.896_y45.396_z-24.465_left"
    pairs = s.split('_')

    x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
    return (x,y,z)
def get_img_matrix(s):
    ori_x, ori_y, ori_z = get_ori_coord(s)
    eye_world = torch.tensor([ori_x, ori_y, ori_z])
    return eye2world_pytroch(eye_world)

def convert_RGB(img):
    width = img.width
    height = img.height

    image = Image.new('RGB', size=(width, height), color=(255, 255, 255))
    image.paste(img, (0, 0), mask=img)

    return image
def get_focal_imgs(idx, scene, return_depth = False):
    
    images_path = os.listdir(data_path)
    images_path = sorted(images_path, key=sort_key)
    views = torch.stack([get_img_matrix(image_path) for image_path in images_path])

    N = depth_num
    N_img = len(images_path)

    N_s = N
    
    
    physical_screen_w = 51.84
    if scene == 'lego_bulldozer':
        scale_physical2world = 0.5/6
        scale_pixel2world = scale_physical2world * physical_screen_w / W
        thickness = thickness_
        Z_world = thickness * scale_physical2world
        z_min = ground = -Z_world/2
        z_max = (ground + Z_world)
        ###
        # z_min = z_min - Z_world / 2
        # z_max = z_max + Z_world / 2
        ###
        W_w = W * scale_pixel2world
        H_w = H * scale_pixel2world
        coord_screen_world = torch.stack([
            torch.Tensor([[z, -W_w/2, H_w/2], [z, W_w/2, H_w/2], [z, -W_w/2, -H_w/2], [z, W_w/2, -H_w/2]
        ]) for z in torch.linspace(z_min, z_max, N_s).tolist()])
    elif scene == 'orchids':
        scale_physical2world = 2/6
        scale_pixel2world = scale_physical2world * physical_screen_w / W
        thickness = -1
        Z_world = 4
        ground = -Z_world*1.2
        coeffient = 0.25
        z_min = ground - coeffient * Z_world
        z_max = (ground + Z_world) + coeffient * Z_world
        W_w = W * scale_pixel2world
        H_w = H * scale_pixel2world
        coord_screen_world = torch.stack([
            torch.Tensor([[z, -W_w/2, H_w/2], [z, W_w/2, H_w/2], [z, -W_w/2, -H_w/2], [z, W_w/2, -H_w/2]
        ]) for z in torch.linspace(z_min, z_max, N_s).tolist()])
    coord_pixel_init = torch.Tensor([(0, 0), (W, 0), (0, H), (W, H)]).view(1,4,2).repeat(N_s,1,1)
    
    coord_src = coord_screen_world.view(N_s, 4, 3, 1).repeat(1, N_img, 1, 1).view(-1, 4, 3)
    view = views[None].repeat(N,1,1,1).view(-1,4,4)
    coord_src_homo = torch.cat([coord_src, torch.ones(N*N_img,4,1)], dim=-1)
    coord_dst = torch.matmul(torch.inverse(view)[:, None], coord_src_homo[..., None]).squeeze(-1)[..., :3] # N 4 3
    u = (-fx*coord_dst[..., [0]]/coord_dst[..., [2]] + W/2)
    v = (fx*coord_dst[..., [1]]/coord_dst[..., [2]] + H/2)
    coord_dst_img = torch.cat([u, v], dim=-1).view(N, N_img, 4, 2)
    imgs = []
    for i in range(0, len(images_path)):
        im = to_tensor(Image.open(data_path+'/'+images_path[i]))
        imgs.append(im)
    imgs = torch.stack(imgs)
    imgs, coord_dst_img = imgs.cuda(), coord_dst_img.cuda()
    
    depth_stack = torch.stack([perspective(img, src.tolist(), dst.tolist()) 
                            for img, src, dst in zip(imgs, coord_dst_img[idx,:], coord_dst_img[idx,60].repeat(N_img,1,1))])
    
    if return_depth:
        return ((coord_screen_world[[-1], 0, 0] - coord_screen_world[:, 0, 0]) / scale_physical2world).tolist(), depth_stack.mean(0)
    return depth_stack.mean(0)

def get_manual_masks():
    masks = []
    for i in range(slice_num):
    # for i in chosen_ls:
        mask = torch.load(os.path.join(mask_path, 'mask{}.pt'.format(i))).cuda()
        # mask = torch.load(os.path.join(mask_path, 'mask{}.pt'.format(20))).cuda()
        masks.append(mask)
    return masks


device = 'cuda:0'

img_path = r"dataset\experiment\depth\center.jpg"
data_path = r"dataset\experiment\depth\depth_imgs"



slice_num = 5   
thickness_ = 12                                                               
to_tensor = T.ToTensor()
pattern_size = (1080, 1920)
pattern_size = torch.Size(pattern_size) # H W
H, W = pattern_size
FOV = math.radians(40)
fx = W/2 / math.tan(FOV/2)
#####################################
mask_path = r'experiment\depth\masks'
depth_num = 75

src_img = Image.open(img_path)
src_img_t = to_tensor(src_img).cuda()
masks = get_manual_masks()

depth_dict = {}

depths = None
with torch.no_grad():

    for j in range(slice_num):
        # depth_dict[j] = []
        psnrs = []
        lap_vars = []
        
        for idx in range(depth_num):
            if depths == None:
                depths, depth_img = get_focal_imgs(idx, scene='lego_bulldozer', return_depth=True)
            else:
                depth_img = get_focal_imgs(idx, scene='lego_bulldozer')
            mse = F.mse_loss(depth_img*masks[j], src_img_t*masks[j])
            psnr = get_PSNR(mse.item(), masks[j])
            psnrs.append(psnr)
        depth_dict[j] = psnrs

print('depths: ', depths)
print('depth_dict: ', depth_dict)
