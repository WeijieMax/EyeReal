from torch.utils.data import Dataset
import torch
import os
from PIL import Image
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
    rt[:3, 3] = eye_world 
    
    return rt
def get_ori_coord(s):
    pairs = s.split('_')

    x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
    return (x,y,z)
   
def get_img_matrix(s):
    ori_x, ori_y, ori_z = get_ori_coord(s)
    eye_world = torch.tensor([ori_x, ori_y, ori_z])
    return eye2world_pytroch(eye_world=eye_world)
class NTFDataset(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, transform=None, data_prefix=None, extra_gt_path=None):
        self.data_prefix = data_prefix
        self.images_path = images_path
        self.images_view = self.views_generation(2)
        self.transform = transform
        if extra_gt_path is not None:
            extra_gt = os.listdir(extra_gt_path)
            extra_gt.sort()
            self.extra_gt = torch.stack([self.transform(Image.open(os.path.join(extra_gt_path, gt))) for gt in extra_gt], dim=1)
        else:
            self.extra_gt = None
            

    def __len__(self):
        return -1

    def __getitem__(self, dataset_idx):
        
        idxs = [dataset_idx * 2, dataset_idx * 2 + 1]

        imgs = list()
        labels = list()
        mask_views = list()
        for idx in idxs:
            if self.data_prefix is not None:
                img = Image.open(os.path.join(self.data_prefix, self.images_path[idx]))
            else:
                img = Image.open(self.images_path[idx])
            
            mask_views.append(get_img_matrix(self.images_path[idx]))

            # RGB为彩色图片，L为灰度图片
            if img.mode != 'RGB':
                raise ValueError("image: {} isn't RGB mode.".format(self.images_path[idx]))
            if self.extra_gt is not None:
                label = self.extra_gt
            else:
                # label = self.images_view[item]
                label = self.images_view[idx%2]

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
            labels.append(label)

        return torch.stack(imgs), torch.stack(labels), torch.stack(mask_views)

    def views_generation(self, num_side, step=1):
    
    
        
        # return torch.cat([view_H, view_W], dim=-1)
        return torch.tensor([
                    [0, -0.5],
                    [0, 0.5]
                ])

    @staticmethod
    def collate_fn(batch):
        
        images, labels,mask_views = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        
        labels = torch.stack(labels, dim=0)
        mask_views = torch.stack(mask_views, dim=0)
        return images, labels,mask_views