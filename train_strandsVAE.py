#!/usr/bin/env python3

# python3 ./train_strandsVAE.py --dataset_path=<PATH_DIFFLOCKS> --exp_info=<name>

import sys
import os
import argparse
from torch.utils.data import DataLoader
from callbacks.callback_utils import *
from models.strand_codec import StrandCodec
import numpy as np
import random
from losses.losses import *
from modules.difflocks.difflocks_utils.general_util import summary
from schedulers.pytorch_warmup.untuned import UntunedLinearWarmup
from modules.difflocks.difflocks_utils.strand_util import compute_dirs
from modules.difflocks.difflocks_utils.general_util import random_quaternions, quaternion_to_matrix
from losses.loss import StrandVAELoss
from tqdm import tqdm
from data_loader.dataloader import DiffLocksDataset
from data_loader.mesh_utils import World2Local

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = True


class HyperParamsStrandVAE:
    def __init__(self):
        self.nr_verts_per_strand=256
        self.use_fourier_space_strands=False
        self.enable_vae = True

        self.nr_iters_to_train=3000000 
        self.lr= 3e-3 #for batch size 10
        self.save_checkpoint=True
        self.save_checkpoint_every_x_epoch=10
        self.with_tensorboard=True
        self.with_visualizer=False
        self.viewer_config_path=os.path.join(SCRIPT_DIR, "./configs/strand_vae_train.toml")

        #####Input
        self.normalize_input=True


        #####Output
        self.decode_type="dir"
        self.scale_init=30.0
        self.nr_verts_per_strand=256
        self.nr_values_to_decode=255
        self.dim_per_value_decoded=3
        

        ###LOSS######
        self.loss_pos_weight=0.5 
        self.loss_dir_weight=1.0
        self.loss_curv_weight=20.0
        self.loss_kl_weight=6e-4 
    



def create_dataloaders(dataset_path):
    difflocks_dataset_train = DiffLocksDataset(dataset_path, 
                                                processed_difflocks_path=None,
                                                train=True, load_rgb_imgs=False, load_full_strands=True,\
                                                load_guide_strands=False, load_interpolated_strands=False, 
                                                load_cam=False,
                                                compute_tbn_full_strands=True,
                                                nr_full_strands_per_hairstyle=20,
                                                check_validity=True,
                                                overfit=False,
                                                train_ratio=0.9)
    difflocks_dataset_test = DiffLocksDataset(dataset_path, 
                                                processed_difflocks_path=None,
                                                train=False, load_rgb_imgs=False, load_full_strands=True,\
                                                load_guide_strands=False, load_interpolated_strands=False, 
                                                load_cam=False,
                                                compute_tbn_full_strands=True,
                                                nr_full_strands_per_hairstyle=20,
                                                check_validity=True,
                                                overfit=False,
                                                train_ratio=0.9)

    loader_train = DataLoader(difflocks_dataset_train, batch_size=10, num_workers=8, shuffle=True, pin_memory=True, persistent_workers=True,
                              prefetch_factor=3)
    loader_test = DataLoader(difflocks_dataset_test, batch_size=10, num_workers=8, shuffle=False, pin_memory=True, persistent_workers=True,
                             prefetch_factor=3)

    return loader_train, loader_test


#transforms the data to a local space, put it on cuda device and reshapes it the way we expect it to be
def prepare_gt_batch(batch, hyperparams, world2local, do_augmentation=False):
    gt_dict = {}

    tbn=batch['full_strands']["tbn"].cuda()
    positions=batch['full_strands']["positions"].cuda()
    root_normal=batch['full_strands']["root_normal"].cuda()

    #get it on local space
    gt_strand_positions, gt_root_normals = world2local(tbn, positions, root_normal)

    #reshape it to be nr_strands, nr_points, dim
    gt_strand_positions=gt_strand_positions.reshape(-1,256,3)

    if do_augmentation:
        nr_strands = gt_strand_positions.shape[0]

        #do some random horizontal flip
        rand_strand_mask=torch.rand(nr_strands, device="cuda")>0.5
        gt_strand_positions[rand_strand_mask,:,0] = -gt_strand_positions[rand_strand_mask,:,0]

        #a bit of rotation do it through quaternions since they allows for linear interpolation which actually does a slerp. If they were rotation matrices I would need to implement slerp
        rotations_q = random_quaternions(nr_strands)
        identity_q = torch.tensor([1, 0, 0, 0], device="cuda").view(1,4).repeat(nr_strands,1)
        #interpolate more towards an identity rotation
        rot_amount=0.1
        rotations_q = rotations_q*rot_amount + identity_q*(1.0-rot_amount)
        rotations = quaternion_to_matrix(rotations_q)
        #rotate positional data  [Nr_strands, 3, 3] x [Nr_strands, nr_points_per_strand, 3]
        rotations = rotations.reshape(nr_strands, 1, 3, 3)
        gt_strand_positions = gt_strand_positions.reshape(nr_strands, -1, 3, 1)
        gt_strand_positions= torch.matmul(rotations, gt_strand_positions)
        gt_strand_positions=gt_strand_positions.reshape(-1,256,3)



    #center the data to be drawn from unit gaussian
    # gt_strand_positions_normalized=whiten_gt_data(gt_strand_positions, normalization_dict, normalization_type="xyz")

    gt_dirs=compute_dirs(gt_strand_positions, append_last_dir=False) #nr_strands,256-1,3
    gt_curv=compute_dirs(gt_dirs, append_last_dir=False) #nr_strands,256-2,3


    gt_dict["strand_positions"]=gt_strand_positions
    gt_dict["strand_directions"]=gt_dirs
    gt_dict["strand_curvatures"]=gt_curv

    return gt_dict


def train(args, hyperparams, loader_train, loader_test, experiment_name, output_training_path):

    cb=create_callbacks(with_tensorboard=hyperparams.with_tensorboard,\
                        with_visualizer=hyperparams.with_visualizer,\
                        viewer_config_path=hyperparams.viewer_config_path,\
                        experiment_name=experiment_name)

    #create phases
    phases= [
        Phase('train', loader_train, grad=True),
        Phase('test', loader_test, grad=False),
    ]

    #model 
    model = StrandCodec(do_vae=hyperparams.enable_vae, 
                        scale_init=hyperparams.scale_init,
                        nr_verts_per_strand=hyperparams.nr_verts_per_strand, nr_values_to_decode=hyperparams.nr_values_to_decode,
                        dim_per_value_decoded=hyperparams.dim_per_value_decoded).to(args.device)
    model = torch.compile(model)

    #misc
    world2local=torch.compile(World2Local())
    loss_computer= torch.compile(StrandVAELoss())
    normalization_dict=loader_train.dataset.get_normalization_data()
    

    # #optimizer
    optimizer = torch.optim.AdamW (model.parameters(), amsgrad=False, lr=hyperparams.lr, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams.nr_iters_to_train)
    scheduler_warmup = UntunedLinearWarmup(optimizer)

    progress_bar = tqdm(range(0, hyperparams.nr_iters_to_train), desc="Training progress")


    is_in_training_loop=True
    while is_in_training_loop:

        for phase in phases:
            model.train(phase.grad)
            if hyperparams.enable_vae:
                model.encoder.do_vae=phase.grad #when testing we don't do any VAE stuff and rather just predict the mean

            cb.phase_started(phase=phase)
            cb.epoch_started(phase=phase)


            #run epoch 
            for batch in iter(phase.loader):
                cb.before_forward_pass(phase=phase)

                #progress
                if phase.grad and phase.iter_nr%100==0 :
                    progress_bar.update(100)

            
                #world_to_local
                with torch.no_grad():
                    gt_dict = prepare_gt_batch(batch, hyperparams, world2local, do_augmentation=phase.grad) 
               

                #forward
                pred_dict, latent_dict = model(gt_dict, hyperparams, normalization_dict)
                
                #loss
                loss_dict = loss_computer(phase, gt_dict, pred_dict, latent_dict, hyperparams)
                loss=loss_dict["loss"]
                

                if torch.isnan(loss):
                    print("found nan")
                    exit()


                #backward
                if phase.grad:
                    optimizer.zero_grad()
                    cb.before_backward_pass()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) 
                    cb.after_backward_pass()
                    optimizer.step()

                    with scheduler_warmup.dampening():
                        lr_scheduler.step()

                
                z_deviation=None
                if "z_deviation" in latent_dict:
                    z_deviation=latent_dict["z_deviation"]


                cb.after_forward_pass(phase=phase, loss=loss, 
                                    loss_pos=loss_dict["loss_pos"], loss_dir=loss_dict["loss_dir"], loss_curv=loss_dict["loss_curv"],
                                    loss_kl=loss_dict["loss_kl"],
                                    gt_cloud=gt_dict["strand_positions"], pred_cloud=pred_dict["strand_positions"],
                                    z_deviation=z_deviation,
                                    z=latent_dict["z"],
                                    z_no_eps=latent_dict["z_no_eps"],
                                    lr=optimizer.param_groups[0]['lr'])

            cb.epoch_ended(phase=phase)
            cb.phase_ended(phase=phase, model=model, hyperparams=hyperparams, experiment_name=experiment_name, output_training_path=output_training_path)

            if phase.grad and phase.iter_nr>=hyperparams.nr_iters_to_train:
                print("Done training!")
                is_in_training_loop=False
                model.save(output_training_path, experiment_name, hyperparams, phase.epoch_nr, info="final")
                exit(1)

        

def main():

    #argparse
    parser = argparse.ArgumentParser(description='Train sdf and color')
    parser.add_argument('--dataset_path', required=True, help='Path to the difflocks dataset to train on')
    parser.add_argument('--exp_info', default="", help='Experiment info string useful for distinguishing one experiment for another')
    parser.add_argument('--device', default="cuda")
    args = parser.parse_args()

    #get the output path which will be at the root of the package 
    hair_forge_root=os.path.dirname(os.path.abspath(__file__))
    output_training_path=os.path.join(hair_forge_root, "out_training")
    os.makedirs(output_training_path, exist_ok=True)


    experiment_name="hair_forge"
    if args.exp_info:
        experiment_name+="_"+args.exp_info


    loader_train, loader_test= create_dataloaders(args.dataset_path)

  
    hyperparams=HyperParamsStrandVAE() 
    train(args, hyperparams, loader_train, loader_test, experiment_name, output_training_path)

    #finished training
    return


if __name__ == '__main__':
    main()
    
