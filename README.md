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

```bash
export HF_ENDPOINT="https://hf-mirror.com"
pip install -U huggingface_hub

huggingface-cli download --resume-download --repo-type dataset WeijieMa/EyeReal --local-dir ./dataset/EyeReal
```

Combine the split files and extract:
```bash
# Combine scene_data parts
cat scene_data.tar.gz.part-a* > scene_data.tar.gz
tar -zxvf scene_data.tar.gz

# Combine uco3d_data parts
cat uco3d_data.tar.gz.part-a* > uco3d_data.tar.gz
tar -zxvf uco3d_data.tar.gz

# Extract other zip files
unzip eval.zip
unzip uco3d_processed_disrupt_train.zip
unzip uco3d_processed_disrupt_val.zip
unzip uco3d_processed_train_sample.zip
unzip uco3d_processed_val_sample.zip
```

### Customized generation

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
For uco3d series, download the uco3d dataset and unzip it into:

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
The model weights can be downloaded from [Huggingface](https://huggingface.co/datasets/WeijieMa/EyeReal/tree/main).
Change the `EyeRealNet_weights` to the path to pretrained_model.pth.
### Inference Based on Coordinates
```bash
# Script: inference_coordinates.py
# - Update `EyeRealNet_weights` with your trained weights.
# - Modify `gaussian_path` to point to the .ply file corresponding to your light field data.
# - Set `mid_coord` to the desired viewpoint position in the world coordinate system (in centimeters).
python inference_coordinates.py
```
### Inference Based on Input Images
Change the `EyeRealNet_weights` to the path to pretrained_model.pth.
```bash
# Script: inference_figures.py
# - Update `EyeRealNet_weights` with your trained weights.
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

## Experiment

First, put the pretrained_model.pth  in the path ./weight/model_ckpts/pretrained_model.pth\
Then, put the eval dataset in the path ./dataset/eval/lego_bulldozer

### Seamless ultrawide visualization


```bash
# data preparation
python experiment\seamless_ultrawide_visualization\data_create.py 
# calculate psnr
python experiment\seamless_ultrawide_visualization\calc_psnr.py
```

### Benchmark for view-segmeted automultiscopy


```bash

python experiment\benchmark_for_view_segmeted_automultiscopy\calc_eye_neighborhood.py

```
### Focal discrimination for depth perceptual continuity


```bash
python experiment\focal_discrimination\calc_focal_stack.py

```

### Benchmark for view-dense automultiscopy (Neural representative)

First, need install some packages

```bash
pip install chainer
# if env has cuda11
pip install cupy-cuda11x
python -m cupyx.tools.install_library --library cudnn --cuda 11.x
```

```bash
# data preparation
python experiment\benchmark_for_view_dense_automultiscopy_neural\data_create_DNN.py
# calculate psnr
python experiment\benchmark_for__view_dense_automultiscopy_neural\calc_DNN.py
#calculate time
python experiment\benchmark_for_view_dense_automultiscopy_neural\time_DNN.py
```

### Benchmark for view-dense automultiscopy (Iterative representative)


```bash
# calculate psnr
python experiment\benchmark_for_view_dense_automultiscopy_iterative\calc_NTF.py
```