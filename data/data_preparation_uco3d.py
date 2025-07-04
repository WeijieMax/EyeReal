import os
import math
import torch
import random
import dataclasses
import torchvision

import numpy as np
from uco3d import (
    GaussianSplats,
    get_all_load_dataset,
)
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from data.render import GaussianRender
from dataset import eye2world_pytroch
from train_eyeReal import init_scene_args
from config.args import get_gaussian_parser
from config.args import get_parser


DEBUG = False

# MLP_ROLE_INDEX = int(os.environ.get('MLP_ROLE_INDEX'))
MLP_ROLE_INDEX = 0

def _truncate_gaussians_outside_sphere(
    splats: GaussianSplats,
    thr: float,
) -> GaussianSplats:
    if splats.fg_mask is None:
        fg_mask = torch.ones_like(splats.means[:, 0], dtype=torch.bool)
    else:
        fg_mask = splats.fg_mask
    centroid = splats.means[fg_mask].mean(dim=0, keepdim=True)
    ok = (splats.means - centroid).norm(dim=1) < thr
    dct = dataclasses.asdict(splats)
    splats_truncated = GaussianSplats(
        **{k: v[ok] for k, v in dct.items() if v is not None}
    )
    return splats_truncated

def _construct_list_of_attributes(splats):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(splats["sh0"].shape[-1]):
        l.append("f_dc_{}".format(i))
    for i in range(splats["shN"].shape[-2:].numel()):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(splats["scales"].shape[-1]):
        l.append("scale_{}".format(i))
    for i in range(splats["quats"].shape[-1]):
        l.append("rot_{}".format(i))
    return l

@torch.no_grad()
def gsplat_to_ply(splats):
    """
    The result ply file can be visualized in standard 3D viewers.
    E.g. in https://antimatter15.com/splat/.
    """
    splats = dataclasses.asdict(splats)
    splats = splats.copy()
    xyz = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = splats["sh0"].detach().flatten(start_dim=1).contiguous().cpu().numpy()
    n = xyz.shape[0]
    if splats.get("shN", None) is None:  # add dummy 0 degree harmonics
        splats["shN"] = torch.zeros(f_dc.shape[0], 1, 3).float()
    else:
        splats["shN"] = splats["shN"].reshape(n, -1, 3)
    f_rest = (
        splats["shN"]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = splats["opacities"].detach().reshape(-1, 1).cpu().numpy()
    scale = splats["scales"].detach().cpu().numpy()
    rotation = splats["quats"].detach().cpu().numpy()
    dtype_full = [
        (attribute, "f4") for attribute in _construct_list_of_attributes(splats)
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation),
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    return PlyData([el])


def load_dataset():
    os.environ["UCO3D_DATASET_ROOT"] = "/path/to/uco3d_unzip"

    dataset = get_all_load_dataset(
        frame_data_builder_kwargs=dict(
            load_gaussian_splats=True,
            gaussian_splats_truncate_background=False,
            apply_alignment=True,
            # --- turn off all other data loading options ---
            load_images=False,
            load_depths=False,
            load_masks=False,
            load_depth_masks=False,
            load_point_clouds=False,
            load_segmented_point_clouds=False,
            load_sparse_point_clouds=False,
            # -----------------------------------------------
        ),
    )


    # sort the sequences based on the reconstruction quality score
    seq_annots = dataset.sequence_annotations()
    sequence_name_to_score = {
        sa.sequence_name: sa.reconstruction_quality.gaussian_splats for sa in seq_annots
    }
    sequence_name_to_score = dict(
        sorted(
            sequence_name_to_score.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    
    return sequence_name_to_score, dataset




def randomize(vertical, orientation, scale_physical2world, R_min=10, R_max=150, phi_min=60, phi_max=120, theta_min=40, theta_max=140):

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

    return (x1, y1, z1), (x2, y2, z2)


def main():
    sequence_name_to_score, dataset = load_dataset()
    
    # some global parameters
    FOV = 40
    num = 20
    R_min=125
    R_max=280
    phi_min=40
    phi_max=80
    theta_min=40
    theta_max=140
    
    truncate_gaussians_outside_sphere_thr = 3.5
    
    parser = get_parser()
    args = parser.parse_args()
    args.vertical = 'y'
    args.orientation = 'xoy'
    args.thickness = 6
    args.scale_physical2world = 0.07
    args.ground_coefficient = -0.5
    # args.ground = -1
    
    # init_scene_args(args=args)

    len_data = len(sequence_name_to_score)
    part = len_data//8
    start_idx = part * MLP_ROLE_INDEX
    end_idx = part * (MLP_ROLE_INDEX+1)
    # sequence_name_to_score = sequence_name_to_score[part * MLP_ROLE_INDEX:part * (MLP_ROLE_INDEX+1)]
    print('start with', part * MLP_ROLE_INDEX, part * (MLP_ROLE_INDEX+1))
    pbar = tqdm(range(part))
    for seqi, seq_name in enumerate(sequence_name_to_score):

        if seqi == start_idx + 151:
            break
        
        if not seqi >= start_idx and seqi < end_idx:
            if seqi % 1000 == 0:
                print('jump', seqi)
            continue 

        pbar.update(1)

        dataset_idx = next(dataset.sequence_indices_in_order(seq_name))
        
    
        try:
            frame_data = dataset[dataset_idx]
        

            file_path = '/fs-computility/ai4chem/maweijie/uco3d_processed_train_sample/{}/'.format(seq_name)
            data_folder = '{}{}_scale_{}_R_{}_{}_FOV_{}_theta_{}_{}_phi_{}_{}'.format(
                seq_name, num, round(args.scale_physical2world,3), 
                R_min, R_max, FOV, theta_min, theta_max, phi_min, phi_max)
            data_path = os.path.join(file_path, data_folder)
            if os.path.exists(data_path):
                continue
            os.makedirs(data_path, exist_ok=True)


            assert seq_name == frame_data.sequence_name
            splats_truncated = _truncate_gaussians_outside_sphere(
                frame_data.sequence_gaussian_splats,
                truncate_gaussians_outside_sphere_thr,
            )
            plydata = gsplat_to_ply(splats_truncated)
            
            render = GaussianRender(
                parser=get_gaussian_parser(),
                sh_degree=3, 
                plydata=plydata,
                white_background=True, 
                FOV=FOV / 180 * math.pi)
            
            
            
            if DEBUG:
                debug_path = 'dataset/scene_data/debug/'
                os.makedirs(debug_path, exist_ok=True)
                all_image_pairs = []
            
            for i in tqdm(range(num)):
                eye1, eye2 = randomize(args.vertical, args.orientation, args.scale_physical2world, R_min, R_max, phi_min, phi_max, theta_min, theta_max)
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
                
                if DEBUG:
                    pair_img = torch.cat([eye1_img, eye2_img], dim=1)
                    all_image_pairs.append(pair_img)
            
            if DEBUG and all_image_pairs:

                all_pairs = torch.cat(all_image_pairs, dim=2)  
                torchvision.utils.save_image(
                    all_pairs,
                    os.path.join(debug_path, f"{seq_name}_all_pairs.jpg")
                )
        except:
            pbar.update(1)
            continue

if __name__ == "__main__":
    main()


