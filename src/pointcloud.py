import os 
import glob
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import re
from scipy.spatial.transform import Rotation

# https://github.com/BonJovi1/Stereo-Reconstruction/blob/master/code.ipynb
# Takes disparity map and returns o3d.geometry.Pointcloud 
def pointcloud_from_disparity(disparity_map, left_rectified, camera_param_left, stereo_baseline, max_distance=3.0):
    camera_focal_length_px = camera_param_left['intrinsics'][0,0]
    image_center_w = camera_param_left['intrinsics'][0,2]
    image_center_h = camera_param_left['intrinsics'][1,2] 
    image_width = camera_param_left['width']
    image_height = camera_param_left['height']

    Q = np.float32([[1, 0,                          0,        -image_center_w],
                    [0, 1,                          0,        -image_center_h], 
                    [0, 0,                          0, camera_focal_length_px], 
                    [0, 0,         -1/stereo_baseline,                      0]])
    
    points = cv2.reprojectImageTo3D(disparity_map, Q)
    mask = np.sum(left_rectified, axis=2)>50
    mask_rgb = np.zeros_like(left_rectified)
    mask_rgb[:,:,0] = mask
    mask_rgb[:,:,1] = mask
    mask_rgb[:,:,2] = mask
    offset = 0
    points = points[offset:image_height+offset,offset:image_width+offset,:]*mask_rgb
    # remove nan points that are behind the camera if it exists
    points_filtered = points.reshape(-1,3)
    colors = left_rectified.reshape(-1,3)
    
    # Mask 1: Points that have positive depth
    mask = points_filtered[:,2]>0
    points_filtered = points_filtered[mask]
    colors = colors[mask]    
    # Mask 2: Points that are not -inf/inf
    mask = ~np.isinf(points_filtered).any(axis=1)
    points_filtered = points_filtered[mask]
    colors = colors[mask]    
    # Mask 3: Points that are within max_distance meters
    mask = points_filtered[:,2]<max_distance
    points_filtered = points_filtered[mask]
    colors = colors[mask]        
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_filtered)
    point_cloud.colors = o3d.utility.Vector3dVector(colors/255)
    return point_cloud

def pointcloud_from_folder(data_path, method, camera_param_left, stereo_baseline, apply_mask=False, max_distance=3.0):
    out_path = Path(os.path.join(data_path, 'output', 'PC_camera_'+method))
    # Check if point clouds are already in the output folder
    if out_path.exists():
        num_files = len([file for file in out_path.iterdir() if file.is_file() if file.suffix == '.ply'])
        expected_num_files = len(glob.glob(os.path.join(data_path, 'output', 'DM_'+method, '*.npy')))
        if num_files == expected_num_files:
            print("Point clouds already exist in output folder")
            return
    out_path.mkdir(parents=True, exist_ok=True)
    
    disparity_map_paths = sorted(glob.glob(os.path.join(data_path, 'output', 'DM_'+method, '*.npy'), recursive=True))
    left_rect_paths = sorted(glob.glob(os.path.join(data_path, 'output', 'LEFT_RECT', '*'), recursive=True))
    if apply_mask:
        mask_paths = sorted(glob.glob(os.path.join(data_path, 'output', 'MASKS', '*.npy'), recursive=True))
    for i in tqdm(range(len(disparity_map_paths))):
    #for (disparity_map_path, left_rect_path) in tqdm(list(zip(disparity_map_paths, left_rect_paths))):
        disparity_map = np.load(disparity_map_paths[i])
        left_rect = cv2.imread(left_rect_paths[i])
        left_rect = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
        if apply_mask:
            masks = np.load(mask_paths[i], allow_pickle=True).item()['masks']
            # Make single mask from multiple masks 
            masks = np.any(masks, axis=0)
            disparity_map = cv2.resize(disparity_map, (masks.shape[1], masks.shape[0]))
            disparity_map = disparity_map*masks            
        point_cloud = pointcloud_from_disparity(disparity_map, left_rect, camera_param_left, stereo_baseline, max_distance)
        numbers = [int(s) for s in re.findall(r'-?\d+?\d*', left_rect_paths[i])]
        out_dir = str(out_path / 'pcd_camera{:04d}.ply'.format(numbers[-1]))
        o3d.io.write_point_cloud(out_dir, point_cloud)

def pointcloud_worldframe(point_cloud, T_world_camera):
    return point_cloud.transform(T_world_camera)
    
def pointcloud_worldframe_folder(data_path, method, frame_change=None):
    if frame_change is None:
        frame_change = np.eye(4)
    out_path = Path(os.path.join(data_path, 'output', 'PC_world_'+method))
    # Check if point clouds are already in the output folder
    if out_path.exists():
        num_files = len([file for file in out_path.iterdir() if file.is_file() if file.suffix == '.ply'])
        expected_num_files = len(glob.glob(os.path.join(data_path, 'output', 'PC_camera_'+method, '*.ply')))
        if num_files == expected_num_files:
            print("Point clouds already exist in output folder")
            return
    out_path.mkdir(parents=True, exist_ok=True)

    pc_cam_paths = sorted(glob.glob(os.path.join(data_path, 'output', 'PC_camera_'+method, '*.ply'), recursive=True))
    tf_paths = sorted(glob.glob(os.path.join(data_path, 'TF', '*.txt'), recursive=True))
    for (pc_cam_path, tf_path) in tqdm(list(zip(pc_cam_paths, tf_paths))):
        pointcloud_camera = o3d.io.read_point_cloud(pc_cam_path)
        cam_pose = np.loadtxt(tf_path)
        cam_pos = cam_pose[:3]
        cam_quat = cam_pose[-4:]
        r_cam = Rotation.from_quat(cam_quat)
        T_world_camera = transformation_from_rotation_translation([r_cam.as_matrix()], [cam_pos])[0]
        T_world_camera = np.matmul(T_world_camera, frame_change)
        pointcloud_world = pointcloud_worldframe(pointcloud_camera, T_world_camera)
        numbers = [int(s) for s in re.findall(r'-?\d+?\d*', pc_cam_path)]
        out_dir = str(out_path / 'pcd_world{:04d}.ply'.format(numbers[-1]))
        o3d.io.write_point_cloud(out_dir, pointcloud_world)

def transformation_from_rotation_translation(rotation, translation):
    transformations = []
    for r, t in zip(rotation,translation):
        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t
        transformations.append(T)
    return np.array(transformations)
