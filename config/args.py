import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Auto3D training and testing')
    parser.add_argument('--FOV', type=float, default=40)
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=1, type=int)

    parser.add_argument("--ntf_iteration", type=int)                 
    parser.add_argument("--ntf_num", type=int, default=200)
    
    parser.add_argument("--N_screen", type=int)
    parser.add_argument("--paper_exp_id", type=int)
    parser.add_argument('--delta_x', type=float, default=0.0)
    parser.add_argument('--delta_y', type=float, default=0.0)
    parser.add_argument('--delta_z', type=float, default=0.0)
    parser.add_argument('--scale_physical2world', type=float, default=1.0)
    parser.add_argument('--physical_width', type=float, default=51.84)
    
    parser.add_argument('--ground_coefficient', type=float, default=None)
    parser.add_argument('--ground', type=float, default=0)
    parser.add_argument('--vertical', type=str)
    parser.add_argument('--orientation', type=str)
    parser.add_argument('--scene', type=str)

    parser.add_argument('--l1_mutex_ratio', type=float, default=0.5)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_dir', type=str, default='./')
    
    parser.add_argument('--train_NTF', action='store_true')
    parser.add_argument('--aux_loss_1', action='store_true')
    parser.add_argument('--aux_loss_2', action='store_true')
    parser.add_argument('--l1_loss', action='store_true')
    parser.add_argument('--l1_mutex', action='store_true')
    parser.add_argument('--aux_weight', type=float, default=1.0)
    parser.add_argument('--thickness', type=float, default=4.0, help='cm')

    parser.add_argument('-n', '--exp_name', type=str, default='debug-lego50-960x540')

    parser.add_argument('--N_channel', type=int, default=2)
    parser.add_argument("--debug", action='store_true') 
    parser.add_argument("--mutex", action='store_true')

    parser.add_argument("--blender", action='store_true')
    parser.add_argument("--single_gpu", action='store_true')
    parser.add_argument("--save_preds", action='store_false')
    parser.add_argument("--aux_loss", action='store_true')
    parser.add_argument("--data_aug", action='store_true')
    parser.add_argument('--aux_ratio', type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--kernel_size5", type=int, default=3)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument('--random_ratio', type=float, default=1)
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')

    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.0002, type=float, help='the initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--T_scale', type=float, default=1.0)
    parser.add_argument('--model_id', default='DisplayFormer', help='name to identify the model')
    parser.add_argument('--output_dir', default='./outputs/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--data_path', default='renders/lego/lego_960_540_2x3_FOV50_60', help='dataset root directory') 
    parser.add_argument('--scenes_path', default='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/maweijie-240108120091/ssd/data/eyeReal', help='scenes root directory') 
    
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--ckpt_weights', default='', )
    parser.add_argument('--extra_ckpt_weights', default='', )
    parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers')
    
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--mlp_W", type=int, default=128)
    parser.add_argument("--mlp_D", type=int, default=1)
    
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')

    parser.add_argument('--object_path', type=str, default='', help='object dataset root directory (e.g., UCO3D)')

    parser.add_argument('--val_root', type=str, default='', help='validation dataset root directory')

    return parser

import argparse
def get_gaussian_parser():
    parser = argparse.ArgumentParser(description='eyeReal')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
