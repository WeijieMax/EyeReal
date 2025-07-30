from PIL import Image
import torch
import os
from torch.utils.data import Dataset
from config.scene_dict import scene_id2key, scene_dict_uco3d, left_0_set, left_1_set, scene_dict
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
# class SceneDataset(Dataset):

#     def __init__(self, scenes_path: str, transform=None, pattern_size=None, train_NTF=False):
        
#         self.N_screen = 3
#         self.H, self.W = pattern_size
        
#         self.scenes_path = scenes_path
#         self.transform = transform
#         self.train_NTF = train_NTF
#         if self.train_NTF:
#             self.images_view = self.views_generation()
#             self.N = -1
#         else:
#             self.N = 0
#             self.scene_counts = []
#             for i in range(len(scene_id2key)):
#                 img_path = os.path.join(self.scenes_path, scene_id2key[i])
#                 self.N += len(os.listdir(img_path)) // 2
#                 self.scene_counts.append(self.N-1)
#             # print(self.scene_counts)

#     def get_ori_coord(self, s):
#         pairs = s.split('_')
#         x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
#         return (x,y,z)
   
#     def get_img_matrix(self, s, vertical, delta):
#         ori_x, ori_y, ori_z = self.get_ori_coord(s)
#         eye_world = torch.tensor([ori_x, ori_y, ori_z])
#         return eye2world_pytroch(vertical, eye_world, delta)

#     def __len__(self):
#         return self.N
    
#     def get_screen_coords_world(self, thickness, scale_physical2world, physical_width, ground, ground_coefficient, orientation, delta):
#         '''
#         thickness: the physical length of whole thickness between screens at both sides
#         '''
        
#         H, W = self.H, self.W
        
#         if physical_width == None:
#             physical_width = 51.84
#         if ground == None:
#             ground = 0
        
#         scale_pixel2world = scale_physical2world * physical_width / W
        
#         Z_world = thickness * scale_physical2world
#         if ground_coefficient != None:
#             z_min = Z_world * ground_coefficient
#         else:
#             z_min = ground
#         # print(f"z_min: {z_min}, type: {type(z_min)}")
#         # print(f"Z_world: {Z_world}, type: {type(Z_world)}")
# 
#         z_max = (z_min + Z_world)
#         W_w = W * scale_pixel2world
#         H_w = H * scale_pixel2world
        
#         if orientation == "xoy":
#             coord_screen_world = torch.stack([
#                 torch.Tensor([[-W_w/2, H_w/2, z], [W_w/2, H_w/2, z], [-W_w/2, -H_w/2, z], [W_w/2, -H_w/2, z]
#             ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
#         elif orientation == "xoz":
#             coord_screen_world = torch.stack([
#                 torch.Tensor([[ -W_w/2,z, H_w/2], [ W_w/2, z,H_w/2], [-W_w/2,z,  -H_w/2], [W_w/2, z, -H_w/2]
#             ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
#         elif orientation == "yox":
#             coord_screen_world = torch.stack([
#                 torch.Tensor([[-H_w/2, -W_w/2, z], [-H_w/2, W_w/2, z], [H_w/2, -W_w/2, z], [H_w/2, W_w/2, z]
#             ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
#         elif orientation == "yoz":
#             coord_screen_world = torch.stack([
#                 torch.Tensor([[z, -W_w/2, H_w/2], [z, W_w/2, H_w/2], [z, -W_w/2, -H_w/2], [z, W_w/2, -H_w/2]
#             ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
#         elif orientation == "zox":
#             coord_screen_world = torch.stack([
#                 torch.Tensor([[H_w/2, z, -W_w/2], [H_w/2, z, W_w/2], [-H_w/2, z, -W_w/2], [-H_w/2, z, W_w/2]
#             ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])


#         for index in range(3):
#             coord_screen_world[..., index] = coord_screen_world[..., index] + delta[index]
            
#         return coord_screen_world


#     def __getitem__(self, ind_):
#         # print("scene_id: ", scene_id)
#         # print("scene_id2key ", scene_id2key)
#         # print("scene_id2key[scene_id]", scene_id2key[scene_id])
# 
#         # scene_id_ = 4
        
#         # print("ind_", ind_)
#         scene_id = bisect.bisect_left(self.scene_counts, ind_)
#         # print("scene_id", scene_id)

#         scene_name = scene_id2key[scene_id]
#         arg_dict = scene_dict[scene_name]
#         delta = torch.tensor([0, 0, 0])
#         delta[0] = arg_dict.get('delta_x') if arg_dict.get('delta_x') else 0
#         delta[1] = arg_dict.get('delta_y') if arg_dict.get('delta_x') else 0
#         delta[2] = arg_dict.get('delta_z') if arg_dict.get('delta_x') else 0
# 
#         coord_screen_world = self.get_screen_coords_world(
#             thickness = arg_dict.get('thickness'), 
#             scale_physical2world = arg_dict.get('scale_physical2world'), 
#             physical_width = arg_dict.get('physical_width'), 
#             ground = arg_dict.get('ground'), 
#             ground_coefficient = arg_dict.get('ground_coefficient'), 
#             orientation = arg_dict.get('orientation'), 
#             delta = delta
#         )
        
#         img_path = os.path.join(self.scenes_path, scene_name)
#         img_names = os.listdir(img_path)
# 
#         img_names = sorted(img_names, key=lambda x: sort_key(x, scene_name))
#         # print("img_names: ", len(img_names))
#         ind = ind_ - (0 if scene_id == 0 else self.scene_counts[scene_id-1]+1)
# 
        
#         left_id = ind * 2
#         right_id = ind * 2 + 1
#         indice = [left_id, right_id]

#         imgs = list()
#         views = list()
#         for idx in indice:

#             img = Image.open(os.path.join(img_path, img_names[idx]))

#             if img.mode == 'RGBA':
#                 img = self.convert_RGB(img)
#             elif img.mode != 'RGB':
#                 raise ValueError("image: {} isn't RGB mode.".format(img_names[idx]))
            
#             if self.train_NTF:
#                 view = self.images_view[idx%2]
#             else:
#                 view = self.get_img_matrix(img_names[idx], arg_dict.get('vertical'), delta)

#             if self.transform is not None:
#                 img = self.transform(img)

#             imgs.append(img)
#             views.append(view)

#         return torch.stack(imgs), torch.stack(views), coord_screen_world
    
#     def views_generation(self):
#         return torch.tensor([[0, -0.5], [0, 0.5]])
    
#     @staticmethod
#     def convert_RGB(img):
#         width = img.width
#         height = img.height

#         image = Image.new('RGB', size=(width, height), color=(255, 255, 255))
#         image.paste(img, (0, 0), mask=img)

#         return image

#     @staticmethod
#     def collate_fn(batch):
#         images, views, coord_screen_world = tuple(zip(*batch))

#         images = torch.stack(images, dim=0)
#         views = torch.stack(views, dim=0)
#         coord_screen_world = torch.stack(coord_screen_world, dim=0)
#         return images, views, coord_screen_world


class SceneDataset(Dataset):

    def __init__(self, scenes_path: str, transform=None, pattern_size=None, train_NTF=False):
        
        self.N_screen = 3
        self.H, self.W = pattern_size
        
        self.scenes_path = scenes_path
        self.scene_list = os.listdir(self.scenes_path)
        
        if 'uco3d' in scenes_path:
            # if use our processed uco3d dataset, you can use the following suffix
            if 'val' in scenes_path:
                self.suffix = '1_scale_0.07_R_125_280_FOV_40_theta_40_140_phi_10_70'
            else:
                self.suffix = '500_scale_0.07_R_125_280_FOV_40_theta_40_140_phi_10_70'
        else:
            self.suffix = ''
        
        self.transform = transform
        self.train_NTF = train_NTF
        if self.train_NTF:
            self.images_view = self.views_generation()
            self.N = -1
        else:
            self.N = len(self.scene_list)

    def get_ori_coord(self, s):
        pairs = s.split('_')
        x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
        return (x,y,z)
   
    def get_img_matrix(self, s, vertical, delta):
        ori_x, ori_y, ori_z = self.get_ori_coord(s)
        eye_world = torch.tensor([ori_x, ori_y, ori_z])
        return eye2world_pytroch(vertical, eye_world, delta)

    def __len__(self):
        if 'val' in self.scenes_path:
            return self.N
        elif 'uco3d' in self.scenes_path:
            return self.N * 100
        else:
            return self.N * 1000
    
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


    def __getitem__(self, scene_id_):
        scene_id = scene_id_ % self.N

        scene_name = self.scene_list[scene_id]
        scene_name_with_suffix = scene_name + self.suffix

        if self.suffix == '':
            scene_name_with_suffix = ''

            arg_dict = scene_dict[scene_name]
            delta = torch.tensor([0, 0, 0])
            delta[0] = arg_dict.get('delta_x') if arg_dict.get('delta_x') else 0
            delta[1] = arg_dict.get('delta_y') if arg_dict.get('delta_x') else 0
            delta[2] = arg_dict.get('delta_z') if arg_dict.get('delta_x') else 0
    
            coord_screen_world = self.get_screen_coords_world(
                thickness = arg_dict.get('thickness'), 
                scale_physical2world = arg_dict.get('scale_physical2world'), 
                physical_width = arg_dict.get('physical_width'), 
                ground = arg_dict.get('ground'), 
                ground_coefficient = arg_dict.get('ground_coefficient'), 
                orientation = arg_dict.get('orientation'), 
                delta = delta
            )
        else:
            arg_dict = scene_dict_uco3d
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
            
        img_path = os.path.join(self.scenes_path, scene_name, scene_name_with_suffix)

        img_names = os.listdir(img_path)
        img_names = sorted(img_names, key=lambda x: sort_key(x))
        img_num = len(img_names) // 2
        ind = random.randint(0, img_num-1)

        
        left_id = ind * 2
        right_id = ind * 2 + 1
        indice = [left_id, right_id]

        imgs = list()
        views = list()
        for idx in indice:

            img = Image.open(os.path.join(img_path, img_names[idx]))

            if img.mode == 'RGBA':
                img = self.convert_RGB(img)
            elif img.mode != 'RGB':
                raise ValueError("image: {} isn't RGB mode.".format(img_names[idx]))
            
            if self.train_NTF:
                view = self.images_view[idx%2]
            else:
                view = self.get_img_matrix(img_names[idx], arg_dict.get('vertical'), delta)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
            views.append(view)

        return torch.stack(imgs), torch.stack(views), coord_screen_world
    
    def views_generation(self):
        return torch.tensor([[0, -0.5], [0, 0.5]])
    
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
        return SceneDataset.collate_fn(batch)
