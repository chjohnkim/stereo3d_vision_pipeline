# Stereo 3D Vision Pipeline

This repository contains essential modules for 3D modeling, including rectification, computing disparity, instance segmentation, converting depth to point clouds, and point cloud registration. The figure below illustrates the input and outputs of the vision pipeline along with the modules.  

![Block Diagram](assets/block_diagram.png)


## Setup Conda Environment
To set up the Conda environment, run the following command:
```bash
conda env create -f conda_environment.yaml
```

## Running
1. Download the sample data from this [link](https://drive.google.com/drive/folders/1gkTlWzwKzGYPoTqLH4RnA-BvrIdYoamD?usp=drive_link)

2. Activate conda environment:
```
conda activate stereo3d
```

3. Execute the Python script:
```
python run_vision_pipeline.py
```
```
Usage: run_vision_pipeline.py [OPTIONS]

Options:
  --data_path TEXT      Path to the data folder
  --method TEXT         Method to calculate disparity: RAFT or SGBM
  --apply_mask BOOLEAN  Apply segmentation mask to point cloud generation
  --help                Show this message and exit.
```

## Input Data Structure
The input data should be organized as follows:
```
stereo3d_vision_pipeline/
├── data/
│   └── tree_n/
│       ├── LEFT/
│       │   ├── left0001.jpg
│       │   ├── left0002.jpg
│       │   └── ...
│       ├── RIGHT/
│       │   ├── right0001.jpg
│       │   ├── right0002.jpg
│       │   └── ...
│       ├── TF/
│       │   ├── transform0001.txt
│       │   ├── transform0002.txt
│       │   └── ...
│       ├── left.pkl
│       ├── right.pkl
│       └── stereo_baseline.txt
```