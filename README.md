# EyeReal

Realtime Glasses-Free 3D Display with Seamless Ultrawide Viewing Range using Deep Learning

## Installation

```bash
# Install torch
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Prepare CUDA
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Install GS
cd lib/GS/submodules
git clone --recursive -b main https://github.com/ashawkey/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
git checkout d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea
git submodule update --init --recursive
cd ../../../..
pip install lib/GS/submodules/diff-gaussian-rasterization
pip install lib/GS/submodules/simple-knn
``` 

# Install other dependencies
```bash
pip install opencv-python tqdm plyfile wandb
```

## Preset before running scripts
```bash
# Windows
set PYTHONPATH=/path/to/EyeReal

# Linux
export PYTHONPATH=/path/to/EyeReal
```

## Generate light field
Our processed data can be downloaded from [Huggingface](https://huggingface.co/datasets/WeijieMa/EyeReal/tree/main).

If you want to process your own data, for .ply files:

```bash
# Use lego_bulldozer as an example, other light fields like it

# First, you should have a .ply file corresponding to the desired light field
# We provide a weight file for lego_bulldozer at weight/gaussian_ply/lego_bulldozer.ply
# You can put any .ply file in the weight/gaussian_ply/

# Next, you should select the viewpoints you want to observe and generate corresponding images
# Set parameters in data/data_preparation.py
# R represents physical world distance in centimeters
# phi indicates the angle between the axis perpendicular to your line of sight, theta denotes the angle between the coordinate axis aligned with the longer edge of the screen and the line of sight's projection on the ground
# Preset parameters in the file define viewpoint selection within a conical space: 
# - Front-facing position of lego_bulldozer
# - ±50° horizontal coverage (left/right)
# - ±30° vertical coverage (up/down)
# After configuring these settings, execute the program
python data/data_preparation.py
python data/data_preparation_disrupt.py # disrupt head pose
```

For uco3d files, download the uco3d dataset and unzip it into:
![files](assets/image.png)

```bash
python data/data_preparation_uco3d.py
python data/data_preparation_uco3d_disrupt.py # disrupt head pose
```


## Training
```bash
# After completing the previous steps, you should have the desired light field data images ready.
# you should modify the paramater --scenes_path and --object_path in train.sh
# Execute the modified script in train.sh the terminal
sh train.sh
```

## Inference
We provide two inference methods: one based on user-specified coordinates and another based on a pair of input images.
The model weights can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1oQXisO1kS3MvihgCm090L-bAtywAspZF?usp=sharing) and put them in weight/model_ckpts.
### Inference Based on Coordinates
```bash
# Script: inference_coordinates.py
# - Update `eyeRealNet_weights` with your trained weights.
# - Modify `gaussian_path` to point to the .ply file corresponding to your light field data.
# - Set `mid_coord` to the desired viewpoint position in the world coordinate system (in centimeters).
python inference_coordinates.py
```
### Inference Based on Input Images
```bash
# Script: inference_figures.py
# - Update `eyeRealNet_weights` with your trained weights.
# - Provide a pair of images for the viewpoints by setting the paths for `data_path`, `left_eye_path`, and `right_eye_path`.
python inference_figures.py
```

## Eval
The data for evaluation and our pretrained model can be downloaded from [Huggingface](https://huggingface.co/datasets/WeijieMa/EyeReal/tree/main).
```bash
# you can run eval\evaluate.py to eval the model
export PYTHONPATH=/path/to/EyeReal

python eval/evaluate.py --ckpt_weights PRETRAINED_MODEL.pth --val_root VAL_ROOT
# E.g.
python eval/evaluate.py --ckpt_weights PRETRAINED_MODEL.pth --val_root /path/to/uco3d_processed_val_sample

```