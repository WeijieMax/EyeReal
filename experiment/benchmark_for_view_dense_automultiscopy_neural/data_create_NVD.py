import os
import re
import torch
import math
import torchvision
from config.args import get_gaussian_parser
from data.render import GaussianRender
from tqdm import tqdm

def eye2world_pytroch(eye_world: torch.Tensor):
    eye_world = eye_world.float()

    vecz = eye_world 
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



def parse_coordinates(filename):
    """Parse viewpoint coordinates (x, y, z) from filename and convert to PyTorch tensor"""
    base = os.path.basename(filename)
    matches = re.findall(r'_x([\d.-]+)_y([\d.-]+)_z([\d.-]+)_', base)
    if matches and len(matches[0]) == 3:
        # Convert coordinates to torch.Tensor
        return torch.tensor([float(matches[0][0]), 
                            float(matches[0][1]), 
                            float(matches[0][2])], dtype=torch.float32)
    else:
        raise ValueError(f"Cannot parse coordinates from filename: {filename}")

def generate_filename(base_dir, prefix, point, index):
    """Generate filename with the required format"""
    x, y, z = point.tolist()
    return os.path.join(
        base_dir, 
        f"{prefix}_x{x:.3f}_y{y:.3f}_z{z:.3f}_{index}.jpg"
    )

def sort_key(s):
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)

def create_data_from_folder(data_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    image_names = os.listdir(data_path)
    image_names = sorted(image_names, key=sort_key)
    N = len(image_names) // 2
    for i in tqdm(range(N)):
        

        # Original file paths
        file1 = os.path.join(data_path, image_names[i*2])
        file2 = os.path.join(data_path, image_names[i*2+1])

        # 1. Extract two viewpoint coordinates (convert to PyTorch tensor)
        eye1 = parse_coordinates(file1)
        eye2 = parse_coordinates(file2)
        # print("Original viewpoint coordinates:")
        # print(f"eye1: {eye1}")
        # print(f"eye2: {eye2}")

        # Get base path and prefix for generating new filenames

        prefix = image_names[i*2].split('_')[0] # File name prefix

        # 2. Calculate the quarter points of line segment eye1-eye2
        AB = eye2 - eye1
        d = torch.norm(AB) / 2  # Quarter distance

        # Calculate 5 points on the line segment (using PyTorch tensor operations)
        points_on_line = [
            eye1 - AB * 0.5,
            eye1,
            eye1 + AB * 0.5,
            eye2,
            eye2 + AB * 0.5,
        ]

        # 3. Calculate perpendicular direction
        A = AB  # Vector A (eye1->eye2)
        B = points_on_line[2]  # Vector B (pointing to midpoint q2)
        C = torch.cross(A, B)  # Vector C = A × B (using PyTorch cross product)

        # Normalization
        norm_C = torch.norm(C)
        if norm_C < 1e-10:  # Handle case when cross product is zero
            alt_vector = torch.tensor([1.0, 0.0, 0.0] if abs(A[0]) < 0.9 else [0.0, 1.0, 0.0])
            C = torch.cross(A, alt_vector)
            norm_C = torch.norm(C)
            if norm_C < 1e-10:
                C = torch.tensor([0.0, 0.0, 1.0])

        unit_C = C / norm_C

        # 4. Generate 25 points and create filenames
        # k_values = [-2, -1, 0, 1, 2]
        k_values = [2, 1, 0, -1, -2]
        generated_files = []

        for i, k in enumerate(k_values):
            for j, point in enumerate(points_on_line):
            

        # for i, point in enumerate(points_on_line):
        #     for j, k in enumerate(k_values):
                offset_point = point + k * d * unit_C
                p_view = eye2world_pytroch(offset_point).cuda()
                p_img = render.render_from_view(p_view)
                # Generate filename with the required format
                file_index = i * len(k_values) + j
                new_file = generate_filename(save_path, prefix, offset_point, file_index)
                
                torchvision.utils.save_image(p_img, new_file)
                
                # generated_files.append(new_file)

        # Output results
        # print("\nGenerated 25 filenames:")
        # for idx, fname in enumerate(generated_files):
        #     print(f"Point {idx+1}: {fname}")




render = GaussianRender(
        parser=get_gaussian_parser(),
        sh_degree=3, 
        gaussians_path='./weight/gaussian_ply/lego_bulldozer.ply',
        white_background=True, FOV=40 / 180 * math.pi)

for i in range(0, 7):

    create_data_from_folder(data_path=r"dataset\eval\lego_bulldozer\lego_bulldozer200_scale_0.083_R_{}_{}_FOV_40_theta_40_140_phi_60_120".format(10+20*i, 30+20*i),
                            save_path=r"dataset\experiment\DNN\lego_bulldozer\lego_bulldozer200_25_scale_0.083_R_{}_{}_FOV_40_theta_40_140_phi_60_120".format(10+20*i, 30+20*i))