
import numpy as np
import pathlib
import click
import yaml
from src import (
    disparity,
    camera_calibration_parser,
    rectification,
    pointcloud,
    multiway_registration,
    detectron,
)

@click.command()
@click.option('--data_path', default='sample_data', help='Path to the data folder')
@click.option('--method', default='RAFT', help='Method to calculate disparity: RAFT or SGBM')
@click.option('--apply_mask', default=False, help='Apply segmentation mask to point cloud generation')
def main(data_path, method, apply_mask):
    frame_change = np.array([[ 0, 1, 0, 0],
                             [-1, 0, 0, 0],
                             [ 0, 0, 1, 0],
                             [ 0, 0, 0, 1]])
    print("############## Loading Camera Parameters ##############")
    data_path = pathlib.Path(data_path).absolute()
    assert data_path.is_dir(), f"Data path {data_path} not found"
    print(f"Data path: {data_path}")
    # Create the output folder if it does not exist
    output_folder = data_path.joinpath('output')
    output_folder.mkdir(exist_ok=True)

    stereo_baseline_path = data_path.joinpath('stereo_baseline.txt')
    left_camera_calibration_path = data_path.joinpath('left.pkl')
    right_camera_calibration_path = data_path.joinpath('right.pkl')
    assert stereo_baseline_path.is_file(), f"stereo_baseline.txt not found in {data_path}"
    assert left_camera_calibration_path.is_file(), f"left.pkl not found in {data_path}"
    assert right_camera_calibration_path.is_file(), f"right.pkl not found in {data_path}"

    stereo_baseline = np.loadtxt(stereo_baseline_path)
    cam_param_left = camera_calibration_parser.parse_pkl(left_camera_calibration_path)
    cam_param_right = camera_calibration_parser.parse_pkl(right_camera_calibration_path)
    
    print("\n############## 1. Rectifying Images ##############")
    rectification.rectify_folder(data_path, cam_param_left, cam_param_right)

    print("\n############## 2. Computing Disparity ##############")
    dp = disparity.disparity(method, data_path)
    dp.generate_folder()

    print("\n############## 3. Instance Segmentation ##############")
    if apply_mask:
        seg = detectron.detectron_predictor()
        seg.segment_from_folder(data_path)
    else:
        print(f"Skipping segmentation mask since apply_mask={apply_mask}")

    print("\n############## 4. Generating Point Cloud ##############")
    pointcloud.pointcloud_from_folder(data_path, method, cam_param_left, stereo_baseline, apply_mask)
    pointcloud.pointcloud_worldframe_folder(data_path, method, frame_change=frame_change)

    print("\n############## 5. Registering Point Clouds ##############")
    mr_cfg_path = pathlib.Path('config/multiway_registration.yaml')
    assert mr_cfg_path.is_file(), f"multiway_registration.yaml not found in {mr_cfg_path}"
    mr_cfg = yaml.safe_load(open(mr_cfg_path, 'r'))

    # Creating a formatted string for the output name
    voxel_size = str(mr_cfg['postprocess']['voxel_size']).replace('.', '')
    nb_points = mr_cfg['postprocess']['radius_outlier_filter']['nb_points']
    radius = str(mr_cfg['postprocess']['radius_outlier_filter']['radius']).replace('.', '')
    out_name = f'voxelSize{voxel_size}_filterPoints{nb_points}_filterRadius{radius}'
    
    # Registering point clouds if it doesn't already exist
    out_path = data_path.joinpath('output', out_name+'_'+method+'.ply') 
    if not out_path.exists(): 
        reg = multiway_registration.multiway_registration(
            data_path, 
            method, 
            voxel_size=mr_cfg['preprocess']['voxel_size'], 
            filter_nb_points=mr_cfg['preprocess']['radius_outlier_filter']['nb_points'], 
            filter_radius=mr_cfg['preprocess']['radius_outlier_filter']['radius'], 
            normal_radius=mr_cfg['preprocess']['estimate_normals']['radius'],
            normal_max_nn=mr_cfg['preprocess']['estimate_normals']['max_nn'],
            apply_filter= not apply_mask)
        if not apply_mask: # Segmentation instances cannot be registered, too sparse
            reg.optimize_pose_graph()
        reg.generate_pointclouds(out_name=out_name, 
                                out_voxel_size=mr_cfg['postprocess']['voxel_size'],
                                out_filter_nb_points=mr_cfg['postprocess']['radius_outlier_filter']['nb_points'],
                                out_filter_radius=mr_cfg['postprocess']['radius_outlier_filter']['radius'])
    
    else: 
        print(f'Point clouds already registered in {out_path}')

if __name__=='__main__':
    main()