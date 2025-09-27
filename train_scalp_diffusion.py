#!/usr/bin/env python3


# NCCL_NVLS_ENABLE=0 accelerate launch train_scalp_diffusion.py --config ./configs/config_scalp_texture_conditional.json --batch-size 4 --mixed-precision bf16 --use-tensorboard --save-checkpoints --compile --save-every 100000 --name scalp_diffusion --dataset_path=<PATH_DIFFLOCKS> --dataset_processed_path=<PATH_DIFFLOCKS_PROCESSED>


#accelerate seems to have some issues which can cause hanging:
# https://github.com/NVIDIA/nccl-tests/issues/216
# https://github.com/huggingface/accelerate/issues/2174#issuecomment-1821295563
# if you have any issues with accelerate hanging, try also adding:NCCL_NVLS_ENABLE=0
# if you need to debug something you can add also TORCH_DISTRIBUTED_DEBUG=INFO NCCL_DEBUG=INFO

import argparse
from copy import deepcopy
from functools import partial
import math
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from modules.difflocks.difflocks_utils.general_util import summary

import accelerate
import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm
import sys
import os
from modules.difflocks.difflocks_utils.vis_util import img_2_pca
import random
from data_loader.dataloader import DiffLocksDataset

import k_diffusion as K

torch._dynamo.config.optimize_ddp=False #for some reason we need this otherwise compile doesnnt work


def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def get_cli_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset_path', required=True, help='Path to the difflocks dataset to train on')
    p.add_argument('--dataset_processed_path', required=True, help='Path to the difflocks processed dataset to train on')
    p.add_argument('--batch-size', type=int, default=8,
                   help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                   help='enable gradient checkpointing')
    p.add_argument('--compile', action='store_true',
                   help='compile the model')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--demo-every', type=int, default=500,
                   help='save a demo grid every this many steps')
    p.add_argument('--end-step', type=int, default=None,
                   help='the step to end training at')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only, disables stratified sampling)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--reset-ema', action='store_true',
                   help='reset the EMA')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                   help='the inference checkpoint to resume from')
    p.add_argument('--save-checkpoints', action='store_true',
                   help='save checkpoints every save-every stps')
    p.add_argument('--save-every', type=int, default=10000,
                   help='save every this many steps')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--use-tensorboard', action='store_true',
                   help='flag to tuse tensorboard for logging scalars and images')
    args = p.parse_args()

    return args

def main():
    args=get_cli_args() 

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    config = K.config.load_config(args.config)
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']
    cross_cond = bool(model_config['cross_cond'])

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(0, 2 ** 32 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        print("seeds[accelerator.process_index]",seeds[accelerator.process_index])
        torch.manual_seed(seeds[accelerator.process_index])
        np.random.seed(seeds[accelerator.process_index])
        random.seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())

    inner_model = K.config.make_model(config)
    inner_model_ema = deepcopy(inner_model)

    print("args.compile",args.compile)
    if args.compile:
        inner_model.compile()
        # inner_model_ema.compile()

    if accelerator.is_main_process:
        print(f'Parameters: {K.utils.n_params(inner_model):,}')


        
    use_tensorboard = args.use_tensorboard
    if use_tensorboard and accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_writer = SummaryWriter("tensorboard_logs/"+args.name)
    

    lr = opt_config['lr'] if args.lr is None else args.lr
    groups = inner_model.param_groups(lr)
    opt = optim.AdamW(groups,
                        lr=lr,
                        betas=tuple(opt_config['betas']),
                        eps=opt_config['eps'],
                        weight_decay=opt_config['weight_decay'])
    

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(opt,
                                  inv_gamma=sched_config['inv_gamma'],
                                  power=sched_config['power'],
                                  warmup=sched_config['warmup'])
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt,
                                      num_steps=sched_config['num_steps'],
                                      decay=sched_config['decay'],
                                      warmup=sched_config['warmup'])
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                  max_value=ema_sched_config['max_value'])
    ema_stats = {}

    
    
    difflocks_dataset_train = DiffLocksDataset(args.dataset_path, 
                                            processed_difflocks_path=args.dataset_processed_path,
                                            train=True, 
                                            load_rgb_imgs=False,
                                            load_scalp_texture=True,
                                            load_latents=True,
                                            latents_type_list=["dinov2"],
                                            load_latents_layers=[
                                                ["final_latent", "cls_token"]
                                            ],
                                            load_cam=True,
                                            subsample_factor=1, #subsamples the RGB img to 384
                                            scalp_texture_resolution=size[0],
                                            load_density_imgs=True,
                                            check_validity=True,
                                            check_validity_processed=True,
                                            do_pedantic_checks=False,
                                            randomly_flip=True,
                                            train_ratio=1.0,
                                            )
    train_set = difflocks_dataset_train


    if accelerator.is_main_process:
        try:
            print(f'Number of items in dataset: {len(train_set):,}')
        except TypeError:
            pass

    image_key = dataset_config.get('image_key', 0)
    num_classes = dataset_config.get('num_classes', 0)
    cond_dropout_rate = dataset_config.get('cond_dropout_rate', 0.1)
    class_key = dataset_config.get('class_key', 1)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    inner_model, inner_model_ema, opt, train_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl)

    # if use_wandb:
        # wandb.watch(inner_model)
    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    state_path = Path(f'{args.name}_state.json')

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unwrap(model.inner_model).load_state_dict(ckpt['model'])
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if args.gns and ckpt.get('gns_stats', None) is not None:
            gns_stats.load_state_dict(ckpt['gns_stats'])
        demo_gen.set_state(ckpt['demo_gen'])

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                      max_value=ema_sched_config['max_value'])
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        unwrap(model.inner_model).load_state_dict(ckpt)
        unwrap(model_ema.inner_model).load_state_dict(ckpt)
        del ckpt


    cfg_scale = 1.

    def make_cfg_model_fn(model):
        def cfg_model_fn(x, sigma, class_cond):
            x_in = torch.cat([x, x])
            sigma_in = torch.cat([sigma, sigma])
            class_uncond = torch.full_like(class_cond, num_classes)
            class_cond_in = torch.cat([class_uncond, class_cond])
            out = model(x_in, sigma_in, class_cond=class_cond_in)
            out_uncond, out_cond = out.chunk(2)
            return out_uncond + (out_cond - out_uncond) * cfg_scale
        if cfg_scale != 1:
            return cfg_model_fn
        return model


    #sample a number of random images from the mdoel 
    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def sample_images(nr_images):
        if accelerator.is_main_process:
            tqdm.write('Sampling...')
        n_per_proc = math.ceil(nr_images / accelerator.num_processes)
        x = torch.randn([accelerator.num_processes, n_per_proc, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device)
        dist.broadcast(x, 0)
        x = x[accelerator.process_index] * sigma_max
        model_fn, extra_args = model_ema, {}
        #Not really relevent for our case currently since we don't have classes
        if num_classes:
            class_cond = torch.randint(0, num_classes, [accelerator.num_processes, n_per_proc], generator=demo_gen).to(device)
            dist.broadcast(class_cond, 0)
            extra_args['class_cond'] = class_cond[accelerator.process_index]
            model_fn = make_cfg_model_fn(model_ema)
        sigmas = K.sampling.get_sigmas_karras(100, sigma_min, sigma_max, rho=7., device=device)
        x_0 = K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=not accelerator.is_main_process)
        x_0 = accelerator.gather(x_0)[:nr_images]
        return x_0
        


    def save():
        accelerator.wait_for_everyone()
        filename = f'{args.name}_{step:08}.pth'
        path_checkpoints_root= os.path.join("./out_training/",args.name)
        os.makedirs(path_checkpoints_root, exist_ok=True)
        filename=os.path.join(path_checkpoints_root,filename)
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        inner_model = unwrap(model.inner_model)
        inner_model_ema = unwrap(model_ema.inner_model)
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))
        # if args.wandb_save_model and use_wandb:
            # wandb.save(filename)
        #save config
        config_path = os.path.join(path_checkpoints_root,"config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)


    losses_since_last_print = []

    do_overfit=False
    if do_overfit:
        overfitted_batch= next(iter(train_dl))
    first_batch=None


    try:
        while True:
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                with accelerator.accumulate(model):
                    if do_overfit:
                        batch=overfitted_batch
                    if first_batch is None:
                        first_batch=batch

                   
                    with torch.no_grad():
                        scalp_texture = batch["scalp_texture"]
                        density_img = batch["density_img"] 
                        # print("density_img mean std", density_img.mean(), density_img.std())
                        #bring the standard deviation of the density image (std=0.5) more in line with the scalp texture which has std=0.3 and also mean to zero
                        density_img=(density_img-0.5)/(0.5/model_config['sigma_data'])
                        density_img=torch.nn.functional.interpolate(density_img, size, mode="nearest")
                        reals=torch.cat([scalp_texture,density_img],1)

           
                    # print("reals is",reals.shape)
                    class_cond, extra_args = None, {}
                    cross_cond = bool(model_config['cross_cond'])
                    if num_classes:
                        class_cond = batch[class_key]
                        drop = torch.rand(class_cond.shape, device=class_cond.device)
                        class_cond.masked_fill_(drop < cond_dropout_rate, num_classes)
                        extra_args['class_cond'] = class_cond
                    if cross_cond:
                        extra_args["latents_dict"]=batch["latents"]
                    noise = torch.randn_like(reals)
                    #TODO possibly disable this?
                    # noise = K.utils.pyramid_noise_like(noise, discount=0.3)

                    do_immiscible_diffusion=True
                    if do_immiscible_diffusion:
                        # Doing assignment on the main process (CPU)
                        with torch.no_grad():
                            gathered_noise = [torch.zeros_like(noise) for _ in range(accelerator.num_processes)] # W * [B, C, H, W]
                            dist.all_gather(gathered_noise, noise)
                            gathered_noise = torch.cat(gathered_noise, dim=0) # [WB, C, H, W]
                            distance = torch.linalg.vector_norm(0.10 * reals.to(torch.float16).flatten(start_dim=1).unsqueeze(1) - 0.10 * gathered_noise.to(torch.float16).flatten(start_dim=1).unsqueeze(0), dim=2) # [B, WB]
                            gathered_distance = [torch.zeros_like(torch.tensor(distance)) for _ in range(accelerator.num_processes)]
                            dist.all_gather(gathered_distance, torch.tensor(distance))
                            

                            if accelerator.is_main_process:
                                # Noise Assignment
                                gathered_distance = torch.cat(gathered_distance, dim=0).cpu().numpy() # [WB, WB]
                                _, col_ind = linear_sum_assignment(gathered_distance)
                                # print("Batch Size =", args.batch_size * accelerator.num_processes)
                                # print("Dist BEFORE assignment =", distance.trace())
                                # print("Dist AFTER assignment =", np.sum(distance[row_ind, col_ind]))
                                # print("Dist Change Rate =", (distance.trace() - np.sum(distance[row_ind, col_ind])) / distance.trace())
                                gathered_noise = gathered_noise[col_ind]

                                for process in range(accelerator.num_processes):
                                    start_idx = args.batch_size * process
                                    end_idx = start_idx + args.batch_size
                                    if process == accelerator.process_index:
                                        noise = gathered_noise[start_idx:end_idx].to(accelerator.device)
                                    else:
                                        dist.send(tensor=gathered_noise[start_idx:end_idx].to(accelerator.device), dst=process)
                            else:
                                dist.recv(tensor=noise, src=0)
                                # Note: main process is normally 0, so here we hardcode.
                            accelerator.wait_for_everyone()

                        #### END IMMISCIBLE DIFFUSION ####
                    # do_noise_offset = True    
                    # noise_offset_mult=0.3
                    # if do_noise_offset:
                    #     # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    #     noise += noise_offset_mult * torch.randn(
                    #         (reals.shape[0], reals.shape[1], 1, 1), device=reals.device
                    #     )

                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([reals.shape[0]], device=device)
                    with K.models.checkpointing(args.checkpointing):
                        losses, singleres_losses, multires_losses, mse_losses = model.loss(reals, noise, sigma, **extra_args)
                    loss = accelerator.gather(losses).mean().item()
                    singleres_loss = accelerator.gather(singleres_losses).mean().item()
                    multires_loss = accelerator.gather(multires_losses).mean().item()
                    mse_loss = accelerator.gather(mse_losses).mean().item()
                    losses_since_last_print.append(loss)
                    accelerator.backward(losses.mean())
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0], reals.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    K.utils.ema_update_dict(ema_stats, {'singleres_loss': singleres_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    K.utils.ema_update_dict(ema_stats, {'multires_loss': multires_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    K.utils.ema_update_dict(ema_stats, {'mse_loss': mse_loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()



                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')

                with torch.no_grad(): 
                    if use_tensorboard and step%50==0 and accelerator.is_main_process:
                        tensorboard_writer.add_scalar('scalp_diffuse/' + '/avg_loss', ema_stats['loss'], step)
                        # tensorboard_writer.add_scalar('scalp_diffuse/' + '/avg_singleres', ema_stats['singleres_loss'], step)
                        tensorboard_writer.add_scalar('scalp_diffuse/' + '/avg_mse_loss', ema_stats['mse_loss'], step)
                        # tensorboard_writer.add_scalar('scalp_diffuse/' + '/avg_multires', ema_stats['multires_loss'], step)
                        tensorboard_writer.add_scalar('scalp_diffuse/' + '/loss', loss, step)
                        tensorboard_writer.add_scalar('scalp_diffuse/' + '/lr', opt.param_groups[0]['lr'], step)
                    if use_tensorboard and step%500==0:
                        #sample a images from the model
                        nr_imgs_sample=4
                        sampled_imgs = sample_images(nr_imgs_sample)
                        if accelerator.is_main_process:
                            sampled_imgs_orig = sampled_imgs
                            #for each img in the batch dimension run pca
                            if model_config["input_channels"]>3:
                                sampled_imgs_pca=[]
                                for i in range(nr_imgs_sample):
                                    img = sampled_imgs[i, ...][None, ...]
                                    img = img[:,0:-1,:,:] #get only the scalp texture part
                                    img_pca = img_2_pca(img)
                                    sampled_imgs_pca.append(img_pca)
                                sampled_imgs=torch.cat(sampled_imgs_pca,0)
                            grid = utils.make_grid(sampled_imgs, nrow=math.ceil(nr_imgs_sample ** 0.5), padding=0)
                            grid=(grid.clamp(-1, 1) + 1) / 2
                            tensorboard_writer.add_image('sampled_img', grid, step)
                            #do the same for density
                            sampled_imgs_density=sampled_imgs_orig[:,-1:,:,:]
                            #gat back in [0,1] range
                            sampled_imgs_density=sampled_imgs_density*2.7 + 0.5
                            grid = utils.make_grid(sampled_imgs_density, nrow=math.ceil(nr_imgs_sample ** 0.5), padding=0)
                            grid=grid.clamp(0, 1)
                            tensorboard_writer.add_image('sampled_img_density', grid, step)

                step += 1

                # if step % args.demo_every == 0:
                #     demo()

               
                if step == args.end_step or (step > 0 and step % args.save_every == 0) and args.save_checkpoints:
                    save()

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    return

            epoch += 1
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
