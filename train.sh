export PYTHONPATH=/path/to/EyeReal


# use all data to train
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 --master_port=12355 \
   train.py --exp_name dim32_15scene_2scale_layer6 \
    --image_height 1080 --image_width 1920 --embed_dim 32 --random_ratio 1 \
    --workers 8 --weight-decay 5e-5 --lr 0.006 -b 4 --epoch 40 --kernel_size 3 \
    --FOV 40 \
    --l1_mutex_ratio 0.5 \
    --aux_loss --aux_ratio 0.3 --aux_weight 10 \
    --l1_mutex --wandb \
    --warmup_epochs 1 --N_screen 3 \
    --use_scene --use_object \
    --scenes_path PATH_TO_PROCESSED_SCENE_DATASET \
    --object_path PATH_TO_PROCESSED_OBJECT_DATASET \
    --use_scene_all --use_object_all \
    --object_suffix 500_scale_0.07_R_125_280_FOV_40_theta_40_140_phi_10_70


# use all scene data to train
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 --master_port=12355 \
   train.py --exp_name dim32_15scene_2scale_layer6 \
    --image_height 1080 --image_width 1920 --embed_dim 32 --random_ratio 1 \
    --workers 8 --weight-decay 5e-5 --lr 0.006 -b 4 --epoch 40 --kernel_size 3 \
    --FOV 40 \
    --l1_mutex_ratio 0.5 \
    --aux_loss --aux_ratio 0.3 --aux_weight 10 \
    --l1_mutex --wandb \
    --warmup_epochs 1 --N_screen 3 \
    --use_scene \
    --scenes_path PATH_TO_PROCESSED_SCENE_DATASET \
    --use_scene_all 

# use all object data to train
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 --master_port=12355 \
   train.py --exp_name dim32_15scene_2scale_layer6 \
    --image_height 1080 --image_width 1920 --embed_dim 32 --random_ratio 1 \
    --workers 8 --weight-decay 5e-5 --lr 0.006 -b 4 --epoch 40 --kernel_size 3 \
    --FOV 40 \
    --l1_mutex_ratio 0.5 \
    --aux_loss --aux_ratio 0.3 --aux_weight 10 \
    --l1_mutex --wandb \
    --warmup_epochs 1 --N_screen 3 \
    --use_object \
    --object_path PATH_TO_PROCESSED_OBJECT_DATASET \
    --use_object_all \
    --object_suffix 500_scale_0.07_R_125_280_FOV_40_theta_40_140_phi_10_70

# use chosen scene data to train
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 --master_port=12355 \
   train.py --exp_name dim32_15scene_2scale_layer6 \
    --image_height 1080 --image_width 1920 --embed_dim 32 --random_ratio 1 \
    --workers 8 --weight-decay 5e-5 --lr 0.006 -b 4 --epoch 40 --kernel_size 3 \
    --FOV 40 \
    --l1_mutex_ratio 0.5 \
    --aux_loss --aux_ratio 0.3 --aux_weight 10 \
    --l1_mutex --wandb \
    --warmup_epochs 1 --N_screen 3 \
    --use_scene \
    --scenes_path PATH_TO_PROCESSED_SCENE_DATASET \
    --choose_scene_names scene_1,scene_2

# use chosen object data to train
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 --master_port=12355 \
   train.py --exp_name dim32_15scene_2scale_layer6 \
    --image_height 1080 --image_width 1920 --embed_dim 32 --random_ratio 1 \
    --workers 8 --weight-decay 5e-5 --lr 0.006 -b 4 --epoch 40 --kernel_size 3 \
    --FOV 40 \
    --l1_mutex_ratio 0.5 \
    --aux_loss --aux_ratio 0.3 --aux_weight 10 \
    --l1_mutex --wandb \
    --warmup_epochs 1 --N_screen 3 \
    --use_object \
    --object_path PATH_TO_PROCESSED_OBJECT_DATASET \
    --object_suffix 500_scale_0.07_R_125_280_FOV_40_theta_40_140_phi_10_70 \
    --choose_object_names object_1,object_2

# use chosen mixed data to train
OMP_NUM_THREADS=8 torchrun --nproc_per_node=8 --master_port=12355 \
   train.py --exp_name dim32_15scene_2scale_layer6 \
    --image_height 1080 --image_width 1920 --embed_dim 32 --random_ratio 1 \
    --workers 8 --weight-decay 5e-5 --lr 0.006 -b 4 --epoch 40 --kernel_size 3 \
    --FOV 40 \
    --l1_mutex_ratio 0.5 \
    --aux_loss --aux_ratio 0.3 --aux_weight 10 \
    --l1_mutex --wandb \
    --warmup_epochs 1 --N_screen 3 \
    --use_scene --use_object \
    --scenes_path PATH_TO_PROCESSED_SCENE_DATASET \
    --object_path PATH_TO_PROCESSED_OBJECT_DATASET \
    --object_suffix 500_scale_0.07_R_125_280_FOV_40_theta_40_140_phi_10_70 \
    --choose_scene_names scene_1,scene_2 \
    --choose_object_names object_1,object_2