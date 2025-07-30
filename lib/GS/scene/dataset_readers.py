#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, pose_rotation=[0,0,0], pose_translation=[0,0,0]):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    positions = rotate_pts(pts=positions, rot_x=pose_rotation[0],rot_y=pose_rotation[1],rot_z=pose_rotation[2],translation=pose_translation)
    
    try:
        colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    except (KeyError, ValueError, TypeError, IndexError) as e:
        print(f"Error in obtaining colors: {e}")
        print("randomized color")
        colors = np.random.rand(*positions.shape)

    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except (KeyError, ValueError, TypeError, IndexError) as e:
        print(f"Error in obtaining normals: {e}")
        print("randomized normal")
        normals = np.random.rand(*positions.shape) * 0
        
    # colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def rotate_pts(pts, rot_x=0., rot_y=0., rot_z=0., translation=[0,0,0]):
    # padding = np.array([[[0,0,0,1]]]).repeat(poses.shape[0] ,0)
    R_z = np.array([[math.cos(np.deg2rad(rot_z)),    -math.sin(np.deg2rad(rot_z)),   0],
                    [math.sin(np.deg2rad(rot_z)),     math.cos(np.deg2rad(rot_z)),   0],
                    [0,                            0,                             1]])
    R_y =  np.array([[math.cos(np.deg2rad(rot_y)), 0, math.sin(np.deg2rad(rot_y))],
                     [ 0           ,   1,             0],
                     [-math.sin(np.deg2rad(rot_y)), 0, math.cos(np.deg2rad(rot_y))]])
    R_x =  np.array([[ 1, 0           ,    0],
                      [ 0, math.cos(np.deg2rad(rot_x)),-math.sin(np.deg2rad(rot_x))],
                      [ 0, math.sin(np.deg2rad(rot_x)), math.cos(np.deg2rad(rot_x))]])
    new_pts=[]
    for pt in pts:
        new_pt=np.dot(R_z, np.dot(R_y, np.dot(R_x, pt)))+np.array(translation)
        # new_pt=np.dot(R_z, np.dot(R_y, np.dot(R_x, pt)))
        new_pts.append(new_pt)
    new_pts=np.array(new_pts)
    
    return new_pts

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    
    # if "china" in path:
    #     ply_path = '/cpfs01/shared/pjlab_lingjun_landmarks/data/fixed_point_clouds/china_museum_fixed.ply'
    # if "wukang" in path:
    #     ply_path = '/cpfs01/shared/pjlab_lingjun_landmarks/data/fixed_point_clouds/wukang_fixed.ply'
    
    pose_rotation=[0,0,0]
    pose_translation=[0,0,0]

    if "desk_bookshelf" in path:
        pose_rotation=[-114, 0, 90]
        pose_translation=[4.8, -2.5, 0]
    if "hotdog" in path:
        pose_rotation=[0, -92, 14] 
        pose_translation=[2.9, 0.04, 0.16]
    if "orchids" in path:
        pose_rotation=[-90, 0, 90]
        pose_translation=[19.16, -1.78, 0]
    if "traffic" in path:
        pose_rotation=[216, -1, -28]
        pose_translation=[-0.36, 0, 2.54]
    if "street" in path:
        pose_rotation=[-116, 0, 114]
        pose_translation=[2.56, 0, 1.46]
    if "truck" in path:
        pose_rotation=[80, 182, -92]
        pose_translation=[0.5, -0.24, 0.1]
    if "desk_bookshelf" in path:
        pose_rotation=[234, 0, 122]
        pose_translation=[2.6, -0.24, 3.1]
    if "room_floor" in path:
        pose_rotation=[-20, -90, 86]
        pose_translation=[0, 0.48, -2.92]
    if "wukang_mansion" in path:
        pose_rotation=[194, 23, -31]
        pose_translation=[0.34, -0.02, 2.02]
    if "shoe_rack" in path:
        pose_rotation=[234,  0, 122]
        pose_translation=[2.6, -0.24, 3.1]
    if "china_art_museum" in path:
        pose_rotation=[234, 4, -61]
        pose_translation=[0.54, 0.33, 0.72]
        
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        print(f"data_path: {path} \npose_rotation: ", pose_rotation, "\npose_translation: ", pose_translation, "\n")
        pcd = fetchPly(ply_path, pose_rotation, pose_translation)
    except:
        pcd = None
    
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    # print("Reading Training Transforms")
    # train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    print("Reading Train Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension)
    test_cam_infos = []
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}