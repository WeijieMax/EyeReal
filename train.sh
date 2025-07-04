export PYTHONPATH=/path/to/EyeReal

OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12355 \
   train_eyeReal_fp32.py --exp_name dim32_15scene_2scale_layer6 \
    --scenes_path PATH_TO_PROCESSED_SCENE_DATASET \
    --object_path PATH_TO_PROCESSED_OBJECT_DATASET \
    --image_height 1080 --image_width 1920 --embed_dim 32 --random_ratio 1 \
    --workers 8 --T_scale 1.0 --weight-decay 5e-5 --lr 0.006 -b 4 --epoch 40 --kernel_size 3 \
    --FOV 40 \
    --l1_mutex_ratio 0.5 \
    --aux_loss --aux_ratio 0.3 --aux_weight 10 \
    --l1_mutex --wandb \
    --warmup_epochs 1 --N_screen 3 \

