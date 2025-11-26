import datetime
import os
import time
import torch
import torch.utils.data
import math
from model.network import EyeRealNet
from torchvision import transforms as T
import data.funcs as funcs
import cv2
from data.sampler import *
import gc
from data.funcs import is_main_process
import warnings
from config.scene_dict import *
from model.loss import get_aux_loss
from data.dataset import CombinedDataset
from data.dataset import EyeRealDataset
from inference_coordinates import *
warnings.filterwarnings('ignore')


def get_choose_subdataset_names(use_scene=False, use_object=False, scenes_path=None, object_path=None, use_scene_all=False, use_object_all=False, choose_scene_names=None, choose_object_names=None):
    choose_scene_names_list = []
    choose_object_names_list = []
    if use_scene:
        if use_scene_all:
            choose_scene_names_list = os.listdir(scenes_path)
        else:
            choose_scene_names_list = choose_scene_names.split(',')
    if use_object:
        if use_object_all:
            choose_object_names_list = os.listdir(object_path)
        else:
            choose_object_names_list = choose_object_names.split(',')
    return choose_scene_names_list, choose_object_names_list

def get_scene_dataset(transform, args, choose_subdataset_names=None):
   

   ds = EyeRealDataset(data_root_path = args.scenes_path, 
                    transform=transform, 
                    pattern_size=(args.image_height, args.image_width),
                    N_screen=args.N_screen,
                    data_mode='scene',
                    suffix='',
                    use_all=args.use_scene_all,
                    choose_subdataset_names=choose_subdataset_names)
   return ds

def get_object_dataset(transform, args, choose_subdataset_names=None):

   ds = EyeRealDataset(data_root_path = args.object_path,
                    transform=transform,
                    pattern_size=(args.image_height, args.image_width),
                    N_screen=args.N_screen,
                    data_mode='object',
                    suffix=args.object_suffix,
                    use_all=args.use_object_all,
                    choose_subdataset_names=choose_subdataset_names)
   return ds

def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
        # T.Normalize(
        #     mean=[0.485, 0.456, 0.406], 
        #     std=[0.229, 0.224, 0.225]),
    ]

    return T.Compose(transforms)

def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def save_from_preds(output, ckpt_path):
    save_path_layer = ckpt_path+'/preditions/layers/'
    save_path_view = ckpt_path+'/preditions/views/'
    os.makedirs(save_path_layer, exist_ok=True)
    os.makedirs(save_path_view, exist_ok=True)

    patterns, preds = output
    
    for i, pred in enumerate(patterns):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path_layer+'layer-'+str(i+1)+'.png', pred)
    
    for i, pred in enumerate(preds):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path_view+'view-'+str(i+1)+'.png', pred)

def save_from_preds_in_wandb(output, caption=''):
    patterns, preds, gts, baseline = output
    
    for i, pred in enumerate(patterns):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        img_wandb = wandb.Image(pred, caption=caption)
        wandb.log({'layer-'+str(i+1): img_wandb})
    
    for i, pred in enumerate(preds):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        img_wandb = wandb.Image(pred, caption=caption)
        wandb.log({'view-'+str(i+1): img_wandb})

    for i, pred in enumerate(gts):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        img_wandb = wandb.Image(pred, caption=caption)
        wandb.log({'view_gt-'+str(i+1): img_wandb})
    
def train_one_epoch(args, model: EyeRealNet, optimizer, data_loader, lr_scheduler, 
                    epoch, print_freq, iterations, save_preds=False, ckpt_path='./'):
    model.train()
    metric_logger = funcs.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', funcs.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    train_PSNR = 0
    total_its = 0
    

    for data in metric_logger.log_every(data_loader, print_freq, header):

        # data = next(data_iter)

        # images : 2 k 3 w h
        images, views, coord_screen_world = data

        images, views, coord_screen_world = images.cuda(non_blocking=True), views.cuda(non_blocking=True), coord_screen_world.cuda(non_blocking=True)
        # images, views = images.cuda(non_blocking=True), views.cuda(non_blocking=True)

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+

        
        patterns = model(images, views, coord_screen_world)
        outs = model.module.get_loss(patterns, gt=images, views=views, coord_screen_world=coord_screen_world, return_preds=save_preds) 
        loss = loss_mse = outs['loss_mse']
        PSNR = outs['PSNR']
        
        if args.mutex:
            loss = outs['loss_mutex']
        elif args.l1_loss:
            loss = outs['loss_l1']
        elif args.l1_mutex:
            # loss = 0.2*outs['loss_mutex'] + 1*outs['loss_l1']
            loss = args.l1_mutex_ratio*outs['loss_mutex'] + (1-args.l1_mutex_ratio)*outs['loss_l1']
        
        if args.aux_loss and epoch < int(args.epochs * args.aux_ratio):
            aux_loss = get_aux_loss(patterns, epoch/int(args.epochs * args.aux_ratio), args.aux_weight)
            loss += aux_loss
        loss.backward()

        grad_norm = get_grad_norm(model.module)
        if epoch > 0:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1, norm_type=2)

        optimizer.step()
        
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        train_PSNR += PSNR
        iterations += 1
        metric_logger.update(loss=loss.item(), PSNR=PSNR, lr=round(optimizer.param_groups[0]["lr"], 8))
        if args.wandb:
            log_dict = {"loss": loss.item(), "loss_mse": loss_mse.item(), "PSNR": PSNR, 
                        "learning_rate": optimizer.param_groups[0]["lr"], "grad_norm": grad_norm}
            if args.aux_loss and epoch < int(args.epochs * args.aux_ratio):
                log_dict['aux_loss'] = aux_loss.item()
            if args.mutex or args.l1_mutex:
                log_dict['loss_mutex'] = outs['loss_mutex'].item()
                del log_dict['loss_mse']
                if args.l1_mutex:
                    log_dict['loss_l1'] = outs['loss_l1'].item()
            wandb.log(log_dict)

        if save_preds:
            patterns = patterns[0].detach().clone()
            preds = outs['preds'][0].detach().clone()
            baseline = images[0].mean(dim=0).detach().clone()
            if total_its % 20 == 0:
                save_from_preds((patterns, preds), ckpt_path)

        # del image, view, preds, loss, data
        total_its += 1
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if args.wandb:
        save_from_preds_in_wandb((patterns, preds, images[0], baseline), caption="epoch: {}".format(epoch))

    return train_loss / total_its, train_PSNR / total_its

def main(args):

    workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 
                   args.workers if not args.debug else 0])
    choose_scene_names_list, choose_object_names_list = get_choose_subdataset_names(
        use_scene=args.use_scene,
        use_object=args.use_object,
        scenes_path=args.scenes_path,
        object_path=args.object_path,
        use_scene_all=args.use_scene_all,
        use_object_all=args.use_object_all,
        choose_scene_names=args.choose_scene_names,
        choose_object_names=args.choose_object_names
    )
    if args.use_scene:
        scene_dataset = get_scene_dataset(get_transform(args=args), args=args, choose_subdataset_names=choose_scene_names_list)
    else:
        scene_dataset = None
    if args.use_object:
        object_dataset = get_object_dataset(get_transform(args=args), args=args, choose_subdataset_names=choose_object_names_list)
    else:
        object_dataset = None


    # Check if at least one dataset is specified
    if scene_dataset is None and object_dataset is None:
        raise ValueError("at least one dataset is required! Please use --use_scene or --use_object parameters")

    # Choose different loading methods based on dataset availability
    if scene_dataset is not None and object_dataset is not None:
        # Both datasets exist
        print("=" * 60)
        print("Using Scene and Object two datasets")
        print("=" * 60)
        
        combined_dataset = CombinedDataset(scene_dataset, object_dataset)
        
        sampler = DualDatasetSampler(
            dataset1=scene_dataset,
            dataset2=object_dataset,
            batch_size=args.batch_size,          
            shuffle=True,
            random_ratio=1.0,      
            drop_last=False
        )
        
        data_loader = torch.utils.data.DataLoader(
            combined_dataset,           
            batch_size=args.batch_size,
            sampler=sampler,           
            collate_fn=CombinedDataset.collate_fn,
            num_workers=workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )
        
        dataset = combined_dataset
        
    elif scene_dataset is not None:
        # Only scene_dataset
        print("=" * 60)
        print("Using Scene dataset")
        print("=" * 60)
        
        
        
        sampler = DistributedRandomSampler(
            scene_dataset,     
            shuffle=True,
            drop_last=False,
            random_ratio=args.random_ratio if hasattr(args, 'random_ratio') else 1.0
        )
        
        data_loader = torch.utils.data.DataLoader(
            scene_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=EyeRealDataset.collate_fn,
            num_workers=workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )
        
        dataset = scene_dataset
        
    else:
        # Only object_dataset
        print("=" * 60)
        print("Using Object dataset")
        print("=" * 60)
        
        

        sampler = DistributedRandomSampler(
            object_dataset,     
            shuffle=True,
            drop_last=False,
            random_ratio=args.random_ratio if hasattr(args, 'random_ratio') else 1.0
        )
        
        data_loader = torch.utils.data.DataLoader(
            object_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=EyeRealDataset.collate_fn,
            num_workers=workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )
        
        dataset = object_dataset



    print(f"Data loader batch size: {len(data_loader)}, dataset size: {len(dataset)}")


    print(len(data_loader), len(dataset))
    assert len(data_loader) > 0, 'len(dataset) must >= batch_size * gpus'
    FOV = args.FOV
    if FOV > math.pi:
        FOV = FOV / 180 * math.pi
    model = EyeRealNet(args=args, FOV=FOV)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])#, find_unused_parameters=True)
    single_model = model.module

    if args.ckpt_weights:
        checkpoint = torch.load(args.ckpt_weights, map_location='cpu',  weights_only=False)
        single_model.load_state_dict(checkpoint['model'])
        resume_epoch = 2
        # args.aux_loss = False
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
    
    params_to_optimize = [
        {"params": [p for p in single_model.parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad)

    
    if args.warmup_epochs > 0:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #  lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)
        lambda x: (x / (len(data_loader) * args.warmup_epochs)) \
            if x < len(data_loader) * args.warmup_epochs \
                else (((1 + math.cos(x * math.pi / ((args.epochs - args.warmup_epochs)*len(data_loader)))) / 2) * (1 - 0.01) + 0.01))
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #  lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)
        lambda x: ((1 + math.cos(x * math.pi / (args.epochs*len(data_loader)))) / 2) * (1 - 0.01) + 0.01)

    start_time = time.time()
    iterations = 0

    exp_name = args.exp_name + '-lr' + str(args.lr) + '-ep' + str(args.epochs) + '-' + time.strftime("%Y-%m-%d-%H:%M:%S")
    ckpt_path = args.output_dir + exp_name
    os.makedirs(ckpt_path, exist_ok=True)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    

    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)

        loss, PSNR = train_one_epoch(args, model, optimizer, data_loader, lr_scheduler, epoch, args.print_freq, iterations, 
                            save_preds=args.save_preds, ckpt_path=ckpt_path)
        PSNR = round(PSNR, 3)
        print('Train PSNR {}'.format(PSNR))

        
        dict_to_save = {
            'model': single_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
            'lr_scheduler': lr_scheduler.state_dict(),
        }


        cur_save_path = os.path.join(ckpt_path, 'model_epoch_{}.pth'.format(str(epoch + 1)))
        print('save to ', cur_save_path)
        funcs.save_on_master(dict_to_save, cur_save_path)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == "__main__":
    from config.args import get_parser
    parser = get_parser()
    args = parser.parse_args()



    funcs.init_distributed_mode(args)

    if not is_main_process():
        args.wandb = False
        args.save_preds = False

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_name, name=args.exp_name, dir=args.wandb_dir, mode='offline')
    
    main(args)

    if args.wandb:
        wandb.finish()
