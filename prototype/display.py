from data.render import GaussianRender
from lib.pykinect2 import PyKinectV2
from lib.pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import ctypes
import torch
import math
from PIL import Image
from model.network import EyeRealNet
from torchvision.transforms.functional import to_pil_image
import torchvision
import warnings
import argparse
from copy import deepcopy as c
from data.dataset import eye2world_pytroch
from train import init_scene_args
from config.args import get_parser, get_gaussian_parser
warnings.filterwarnings('ignore')



# Get depth image, default size 424x512
def get_last_depth(kinect: PyKinectRuntime.PyKinectRuntime):
    frame = kinect.get_last_depth_frame()
    frame = frame.astype(np.uint8)
    dep_frame = np.reshape(frame, [kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width])
    return cv2.cvtColor(dep_frame, cv2.COLOR_GRAY2RGB)

# Get RGB image, 1080x1920x4
def get_last_rbg(kinect: PyKinectRuntime.PyKinectRuntime):
    frame = kinect.get_last_color_frame()
    return np.reshape(frame, [kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4])[:, :, 0:3]

def color_2_camera(kinect, depth_frame_data, camera_space_point, as_array=False):
    """
    :param kinect: Class for main file
    :param depth_frame_data: kinect._depth_frame_data
    :param camera_space_point: _CameraSpacePoint structure from PyKinectV2
    :param as_array: returns frame as numpy array
    :return: returns mapped color frame to camera space
    """
    color2world_points_type = camera_space_point * int(1920 * 1080)
    color2world_points = ctypes.cast(color2world_points_type(), ctypes.POINTER(camera_space_point))
    kinect._mapper.MapColorFrameToCameraSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080), color2world_points)
    pf_csps = ctypes.cast(color2world_points, ctypes.POINTER(ctypes.c_float))
    data = np.ctypeslib.as_array(pf_csps, shape=(1080, 1920, 3))
    if not as_array:
        return color2world_points
    else:
        return data
    
def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def visualize_eye(frame, faces, thickness=2, color=(0, 255, 0)):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 255), thickness)
            x_left, y_left, x_right, y_right = coords[4], coords[5], coords[6], coords[7]
            cv2.line(frame, (x_left - 10, y_left), (x_left + 10, y_left), color, thickness)
            cv2.line(frame, (x_left, y_left - 10), (x_left, y_left + 10), color, thickness)
            cv2.line(frame, (x_right - 10, y_right), (x_right + 10, y_right), color, thickness)
            cv2.line(frame, (x_right, y_right - 10), (x_right, y_right + 10), color, thickness)

def set_coord_screen_world(coords_world, rot_angle):

    rot_matrix = torch.FloatTensor([
        [math.cos(rot_angle), 0, math.sin(rot_angle)],
        [0, 1, 0],
        [-math.sin(rot_angle), 0, math.cos(rot_angle)],
    ]).to(coords_world.device)
    
    coord_screen_world_T = torch.transpose(coords_world, 0, 1)
    coords_world = torch.matmul(rot_matrix, coord_screen_world_T).transpose(0, 1)
    return coords_world

def kinect2view(coords: torch.Tensor, kinect2world: torch.Tensor, scale=1, vertical='z'):
    # coords.shape N, 3; kinect2world.shape 4, 4
    N, _ = coords.shape
    coords = coords.nan_to_num().cuda()
    coords = torch.cat([coords, torch.ones(N, 1).cuda()], dim=1).unsqueeze(-1)
    kinect2world = kinect2world.unsqueeze(0).repeat(N, 1, 1)
    coords_world = torch.bmm(kinect2world, coords).squeeze(-1)[:, :3]
    coords_world = coords_world * scale
    views = torch.stack([eye2world_pytroch(vertical=vertical, eye_world=coords_w) 
                         for coords_w in coords_world])
    return views

def preprocess(img, transform):
    img = to_pil_image(img)
    if img.mode == 'RGBA':
        width = img.width
        height = img.height

        image = Image.new('RGB', size=(width, height), color=(255, 255, 255))
        image.paste(img, (0, 0), mask=img)

        return image
    return transform(img)


def main():
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

    detector = cv2.FaceDetectorYN.create(
        './weights/face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.7,
        0.3,
        5000
    )
    face_reduce = 3
    W_face = kinect.color_frame_desc.Width // face_reduce
    H_face = kinect.color_frame_desc.Height // face_reduce
    detector.setInputSize((W_face, H_face))
    FOV = 40 / 180 * math.pi
    
    render = GaussianRender(
        parser=get_gaussian_parser(),
        sh_degree=3, 
        gaussians_path='./weight/gaussian_ply/lego_bulldozer.ply',
        white_background=True, FOV=FOV)
 
    # calibated matrix
    kinect2world = np.array([
        [0.699976, 2.09768, 7.81822, -1.90611], 
        [8.09414, -0.276962, -0.650369, 0.166057],
        [0.0985957, 7.8446, -2.11359, 3.31142], 
        [0, 0, 0, 1]
    ])
    kinect2world = torch.from_numpy(kinect2world).float().cuda()

    args = get_parser().parse_args()
    args.scene = 'lego_bulldozer'
    init_scene_args(args=args)

    model = EyeRealNet(args=args, FOV=FOV)
    model.load_state_dict(torch.load(r"./weight/model_ckpts/lego_bulldozer.pth", map_location='cpu')['model'])
    model = model.cuda()
    model.eval()

    cv2.namedWindow('layer1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('layer2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('layer3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('layer1', 1920, 1080)
    cv2.resizeWindow('layer2', 1920, 1080)
    cv2.resizeWindow('layer3', 1920, 1080)
    cv2.moveWindow('layer1', -1920, 0)
    cv2.moveWindow('layer2', 0, 1080)
    cv2.moveWindow('layer3', 1920, 0)
    screen_resizer = torchvision.transforms.Resize((1080, 1920))
    
    tm = cv2.TickMeter()
    tm_kinect = cv2.TickMeter()
    tm_face = cv2.TickMeter()
    tm_render = cv2.TickMeter()
    tm_model = cv2.TickMeter()
    tm_show = cv2.TickMeter()

    iter_count = 0

    while True:
        if kinect.has_new_depth_frame():
            tm.reset()
            tm_kinect.reset()
            tm_face.reset()
            tm_render.reset()
            tm_model.reset()
            tm_show.reset()
            tm.start()
            tm_kinect.start()
            last_frame = get_last_rbg(kinect)
            camera_coords = color_2_camera(kinect, kinect._depth_frame_data, PyKinectV2._CameraSpacePoint, as_array=True)
            tm_kinect.stop()
            camera_coords = torch.from_numpy(camera_coords).nan_to_num()
            tm_face.start()
            # faces = detector.detect(last_frame)
            faces = detector.detect(cv2.resize(last_frame, (W_face, H_face)))
            tm_face.stop()
            
            if faces[1] is not None:
                for idx, face in enumerate(faces[1]):
                    coords = face[:-1].astype(np.int32)
                    coords[4:8] = coords[4:8] * face_reduce
                    x_left, y_left, x_right, y_right = coords[4], coords[5], coords[6], coords[7]
                    try:
                        coords_left = camera_coords[y_left, x_left]
                        coords_right = camera_coords[y_right, x_right]
                    except:
                        coords_left = coords_right = torch.tensor([torch.nan, torch.nan, torch.nan])
                    continue
            else:
                coords_left = coords_right = torch.tensor([torch.nan, torch.nan, torch.nan])
            
            if torch.isnan(coords_left).any() or torch.isnan(coords_right).any() or \
               (coords_left.abs() >= 1e6).any() or (coords_right.abs() >= 1e6).any():
                continue

            coords = torch.stack([coords_left, coords_right])
            views = kinect2view(coords, kinect2world, scale=1, vertical=args.vertical)
            
            if torch.isnan(views).any():
                continue

            tm_render.start()
            images = torch.stack([render.render_from_view(c(view)).clamp(0, 1) for view in views])
            
            images = images.cuda(non_blocking=True)[None]
            views = views.cuda(non_blocking=True)[None]
            tm_render.stop()

            tm_model.start()
            with torch.no_grad():
                patterns = model(images, views, FOV=FOV)[0] 
            tm_model.stop()

            patterns = screen_resizer(patterns)
            patterns = patterns.mul(255).add(0.5).clamp(0, 255).permute(0, 2, 3, 1).type(torch.uint8).cpu().numpy()

            tm_show.start()
            cv2.imshow('layer1', cv2.cvtColor(np.rot90(patterns[2], k=2), cv2.COLOR_RGB2BGR))
            cv2.imshow('layer2', cv2.cvtColor(np.rot90(patterns[1], k=2), cv2.COLOR_RGB2BGR))
            cv2.imshow('layer3', cv2.cvtColor(np.rot90(patterns[0], k=2), cv2.COLOR_RGB2BGR))
            tm_show.stop()
            tm.stop()
        iter_count += 1
        print('iter:', iter_count,
            '\ttotal:', round(tm.getTimeSec(), 4),'\tFPS:', round(tm.getFPS(), 4), 
            '\tkinect:', round(tm_kinect.getTimeSec(), 4),
            '\tface:', round(tm_face.getTimeSec(), 4),
            '\trender:', round(tm_render.getTimeSec(), 4),
            '\tmodel:', round(tm_model.getTimeSec(), 4),
            '\tshow:', round(tm_show.getTimeSec(), 4), end='\r')

        if (iter_count > 3000) or (cv2.waitKey(1) and 0xff == ord('q')):
            break
    
    cv2.destroyAllWindows()
    kinect.close()


if __name__ == '__main__':
    main()