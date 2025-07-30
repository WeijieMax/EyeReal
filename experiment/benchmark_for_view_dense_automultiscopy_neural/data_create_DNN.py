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
    """从文件名中解析出视点坐标(x, y, z)并转换为PyTorch张量"""
    base = os.path.basename(filename)
    matches = re.findall(r'_x([\d.-]+)_y([\d.-]+)_z([\d.-]+)_', base)
    if matches and len(matches[0]) == 3:
        # 将坐标转换为torch.Tensor
        return torch.tensor([float(matches[0][0]), 
                            float(matches[0][1]), 
                            float(matches[0][2])], dtype=torch.float32)
    else:
        raise ValueError(f"无法从文件名解析坐标: {filename}")

def generate_filename(base_dir, prefix, point, index):
    """生成符合格式要求的文件名"""
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
        

        # 原始文件路径
        file1 = os.path.join(data_path, image_names[i*2])
        file2 = os.path.join(data_path, image_names[i*2+1])

        # 1. 提取两个视点坐标 (转换为PyTorch张量)
        eye1 = parse_coordinates(file1)
        eye2 = parse_coordinates(file2)
        # print("原始视点坐标:")
        # print(f"eye1: {eye1}")
        # print(f"eye2: {eye2}")

        # 获取基础路径和前缀用于生成新文件名

        prefix = image_names[i*2].split('_')[0] # 文件名前缀

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
        generated_files = []

        for i, k in enumerate(k_values):
            for j, point in enumerate(points_on_line):
            

        # for i, point in enumerate(points_on_line):
        #     for j, k in enumerate(k_values):
                offset_point = point + k * d * unit_C
                p_view = eye2world_pytroch(offset_point).cuda()
                p_img = render.render_from_view(p_view)
                # 生成符合格式的文件名
                file_index = i * len(k_values) + j
                new_file = generate_filename(save_path, prefix, offset_point, file_index)
                
                torchvision.utils.save_image(p_img, new_file)
                
                # generated_files.append(new_file)

        # 输出结果
        # print("\n生成的25个文件名:")
        # for idx, fname in enumerate(generated_files):
        #     print(f"点{idx+1}: {fname}")




render = GaussianRender(
        parser=get_gaussian_parser(),
        sh_degree=3, 
        gaussians_path='./weight/gaussian_ply/lego_bulldozer.ply',
        white_background=True, FOV=40 / 180 * math.pi)

for i in range(0, 7):

    create_data_from_folder(data_path=r"dataset\eval\lego_bulldozer\lego_bulldozer200_scale_0.083_R_{}_{}_FOV_40_theta_40_140_phi_60_120".format(10+20*i, 30+20*i),
                            save_path=r"dataset\experiment\DNN\lego_bulldozer\lego_bulldozer200_25_scale_0.083_R_{}_{}_FOV_40_theta_40_140_phi_60_120".format(10+20*i, 30+20*i))