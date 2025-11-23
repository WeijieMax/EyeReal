from PIL import Image
import torch
import os
from torch.utils.data import Dataset
from config.scene_dict import scene_id2key, object_dict, left_0_set, left_1_set, scene_dict
import random
import bisect
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
def sort_key(s):
    # left/right eye order affects model training, need to be fixed
    # "pair0_x5.896_y45.396_z-24.465_left.jpg"
    # pair1717_x2.712_y0.103_z0.381_0.jpg
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction_ = pairs[-1].split('.')[0]
    if direction_ == 'left':
        direction = 0
    elif direction_ == 'right':
        direction = 1
    else:
        direction = int(direction_)     # default: left=0, right=1

    return (index, direction)

class EyeRealDataset(Dataset):

    def __init__(self, data_root_path: str, transform=None, pattern_size=None, N_screen=3, 
    data_mode='scene', suffix='', use_all=False, choose_subdataset_names=None):
        
        self.N_screen = N_screen
        self.H, self.W = pattern_size
        
        self.data_root_path = data_root_path
        self.data_mode = data_mode
        self.suffix = suffix
        self.use_all = use_all
        
        self.data_paths = self.get_data_paths(choose_subdataset_names)
        
        self.transform = transform
        

    def get_data_paths(self, choose_subdataset_names=None):
        self.subdataset_names = choose_subdataset_names
        self.sub_dataset_num = len(self.subdataset_names)
        self.data_paths = []
        self.subdataset_image_num = []
        self.N = 0
        for subdataset_name in self.subdataset_names:
            if self.data_mode == 'scene':
                self.data_paths.append(os.path.join(self.data_root_path, subdataset_name))
                subdataset_num = len(os.listdir(self.data_paths[-1])) // 2
                self.N += subdataset_num
                self.subdataset_image_num.append(self.N)
            elif self.data_mode == 'object':
                self.data_paths.append(os.path.join(self.data_root_path, subdataset_name, subdataset_name + self.suffix))
                subdataset_num = len(os.listdir(self.data_paths[-1])) // 2
                self.N += subdataset_num
                self.subdataset_image_num.append(self.N)
            else:
                raise ValueError(f"Invalid data mode: {self.data_mode}")
                
        return self.data_paths

    def get_ori_coord(self, s):
        pairs = s.split('_')
        x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
        return (x,y,z)
   
    def get_img_matrix(self, s, vertical, delta):
        ori_x, ori_y, ori_z = self.get_ori_coord(s)
        eye_world = torch.tensor([ori_x, ori_y, ori_z])
        return eye2world_pytroch(vertical, eye_world, delta)

    def __len__(self):
        return self.N
    
    def get_screen_coords_world(self, thickness, scale_physical2world, physical_width, ground, ground_coefficient, orientation, delta):
        '''
        thickness: the physical length of whole thickness between screens at both sides
        '''
        
        H, W = self.H, self.W
        
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
        z_min = z_min - Z_world / 2
        z_max = z_max + Z_world / 2
        W_w = W * scale_pixel2world
        H_w = H * scale_pixel2world
        
        if orientation == "xoy":
            coord_screen_world = torch.stack([
                torch.Tensor([[-W_w/2, H_w/2, z], [W_w/2, H_w/2, z], [-W_w/2, -H_w/2, z], [W_w/2, -H_w/2, z]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
        elif orientation == "xoz":
            coord_screen_world = torch.stack([
                torch.Tensor([[ -W_w/2,z, H_w/2], [ W_w/2, z,H_w/2], [-W_w/2,z,  -H_w/2], [W_w/2, z, -H_w/2]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
        elif orientation == "yox":
            coord_screen_world = torch.stack([
                torch.Tensor([[-H_w/2, -W_w/2, z], [-H_w/2, W_w/2, z], [H_w/2, -W_w/2, z], [H_w/2, W_w/2, z]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
        elif orientation == "yoz":
            coord_screen_world = torch.stack([
                torch.Tensor([[z, -W_w/2, H_w/2], [z, W_w/2, H_w/2], [z, -W_w/2, -H_w/2], [z, W_w/2, -H_w/2]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
        elif orientation == "zox":
            coord_screen_world = torch.stack([
                torch.Tensor([[H_w/2, z, -W_w/2], [H_w/2, z, W_w/2], [-H_w/2, z, -W_w/2], [-H_w/2, z, W_w/2]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])


        for index in range(3):
            coord_screen_world[..., index] = coord_screen_world[..., index] + delta[index]
            
        return coord_screen_world


    def __getitem__(self, ind_):
        subdataset_ind = bisect.bisect_right(self.subdataset_image_num, ind_)
        subdataset_name = self.subdataset_names[subdataset_ind]
        subdataset_data_path = self.data_paths[subdataset_ind]
        data_ind = ind_ - self.subdataset_image_num[subdataset_ind-1] if subdataset_ind > 0 else 0

        if self.data_mode == 'scene':
            arg_dict = scene_dict[subdataset_name]
        else:
            arg_dict = object_dict
        delta = torch.tensor([0, 0, 0])
        delta[0] = arg_dict.get('delta_x') if arg_dict.get('delta_x') else 0
        delta[1] = arg_dict.get('delta_y') if arg_dict.get('delta_x') else 0
        delta[2] = arg_dict.get('delta_z') if arg_dict.get('delta_x') else 0

        coord_screen_world = self.get_screen_coords_world(
            thickness = arg_dict.get('thickness'), 
            scale_physical2world = arg_dict.get('scale_physical2world'), 
            physical_width = arg_dict.get('physical_width'), 
            ground = arg_dict.get('ground', None), 
            ground_coefficient = arg_dict.get('ground_coefficient'), 
            orientation = arg_dict.get('orientation'), 
            delta = delta
        )
        
        img_names = os.listdir(subdataset_data_path)
        img_names = sorted(img_names, key=lambda x: sort_key(x))
        
        left_id = data_ind * 2
        right_id = data_ind * 2 + 1
        indice = [left_id, right_id]

        imgs = list()
        views = list()
        for idx in indice:

            img = Image.open(os.path.join(subdataset_data_path, img_names[idx]))

            if img.mode == 'RGBA':
                img = self.convert_RGB(img)
            elif img.mode != 'RGB':
                raise ValueError("image: {} isn't RGB mode.".format(img_names[idx]))
            
            
            view = self.get_img_matrix(img_names[idx], arg_dict.get('vertical'), delta)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
            views.append(view)

        return torch.stack(imgs), torch.stack(views), coord_screen_world
    
    
    
    @staticmethod
    def convert_RGB(img):
        width = img.width
        height = img.height

        image = Image.new('RGB', size=(width, height), color=(255, 255, 255))
        image.paste(img, (0, 0), mask=img)

        return image

    @staticmethod
    def collate_fn(batch):
        images, views, coord_screen_world = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        views = torch.stack(views, dim=0)
        coord_screen_world = torch.stack(coord_screen_world, dim=0)
        return images, views, coord_screen_world

class CombinedDataset(Dataset):
    """Combined dataset class that works with DualDatasetSampler"""
    
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(dataset1)
        self.len2 = len(dataset2)


        
    def __len__(self):
        return self.len1 + self.len2
    
    def __getitem__(self, index):
        if index < self.len1:
            # Sample from dataset1
            return self.dataset1[index]
        else:
            # Sample from dataset2, using modulo for index to handle repeated sampling
            idx2 = (index - self.len1) % self.len2

            return self.dataset2[idx2]
    
    @staticmethod
    def collate_fn(batch):
        # Use dataset1's collate_fn (assuming both datasets have the same collate_fn)
        return EyeRealDataset.collate_fn(batch)
