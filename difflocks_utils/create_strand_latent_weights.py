#!/usr/bin/env python3

#when training the diffusion model we apply a L2 loss on all the channels of the scalp texture. However some of the channels are essentially noise. 
#This is due to the fact that some dimensions of the strand vae encode very little information and don't signficantly modify the strand shape. 
#here we check what is the change in position, direction and curvature when changing each of the dimensions of the latent and we use this delta change as a weight for our diffusion model to downweight certain channels



#python3 ./create_strand_latent_weights.py --checkpoint_path <STAND_CODEC_CHECKPOINT.pt>




import argparse
import torch
from models.strand_codec import StrandCodec
from difflocks_utils.strand_util import compute_dirs
from losses.loss import StrandVAELoss
import json
from data_loader.dataloader import DiffLocksDataset
from data_loader.mesh_utils import world_to_tbn_space




torch.set_grad_enabled(False)


#transforms the data to a local space, put it on cuda device and reshapes it the way we expect it to be
def prepare_gt_batch(batch):
    gt_dict = {}

    tbn=batch['full_strands']["tbn"].cuda()
    positions=batch['full_strands']["positions"].cuda()
    root_normal=batch['full_strands']["root_normal"].cuda()

    #get it on local space
    gt_strand_positions, gt_root_normals = world_to_tbn_space(tbn, 
                                                              positions, 
                                                              root_normal)
    gt_strand_positions=gt_strand_positions.cuda()

    #reshape it to be nr_strands, nr_points, dim
    gt_strand_positions=gt_strand_positions.reshape(-1,256,3)

    gt_dirs=compute_dirs(gt_strand_positions, append_last_dir=False) #nr_strands,256-1,3


    gt_dict["strand_positions"]=gt_strand_positions
    gt_dict["strand_directions"]=gt_dirs


    return gt_dict


class HyperParamsStrandVAE:
    def __init__(self):
        #####Output
        #####decode dir######
        self.decode_type="dir"
        self.scale_init=30.0
        self.nr_verts_per_strand=256
        self.nr_values_to_decode=255
        self.dim_per_value_decoded=3


        ###LOSS######
        #these are the same values that were used to train the strand vae
        self.loss_pos_weight=0.5 ##FOR LAMBDA
        self.loss_dir_weight=1.0
        self.loss_curv_weight=20.0
        self.loss_kl_weight=0.0



def main():

    #argparse
    parser = argparse.ArgumentParser(description='Get the weights of each dimensions after training a strand VAE')
    parser.add_argument('--checkpoint_path', required=True, help='Path to the strandVAE checkpoint')
    args = parser.parse_args()

    path_strand_vae_model=args.checkpoint_path
    

    hyperparams=HyperParamsStrandVAE() 


    normalization_dict=DiffLocksDataset.get_normalization_data()

    model = StrandCodec(do_vae=False, 
                        decode_type="dir",
                        scale_init=30.0,
                        nr_verts_per_strand=256, nr_values_to_decode=255,
                        dim_per_value_decoded=3).cuda()
    model.load_state_dict(torch.load(path_strand_vae_model))
    model = torch.compile(model)



    #latent of dimension 64 and get GT which is the mean strand
    latent=torch.zeros(1,64).cuda()
    pred_dict = model.decoder(latent, None, normalization_dict)
    pred_points=pred_dict["strand_positions"]
    gt_strand=pred_points
    print("gt_strand",gt_strand.shape)

    
    #make loss function
    loss_computer= StrandVAELoss()


    #for each dimension change it by 0.5 and check the error towards the mean strand (GT)
    loss_per_dim=[]
    for i in range(64):
        latent=torch.zeros(1,64).cuda()
        latent[:,i]=0.8
        pred_dict = model.decoder(latent, None, normalization_dict)
        pred_points=pred_dict["strand_positions"]

        #make dicts
        gt_dict={"strand_positions": gt_strand}
        pred_dict={"strand_positions": pred_points}
        latent_dict={}

        #loss
        loss_dict = loss_computer(None, gt_dict, pred_dict, latent_dict, hyperparams)
        loss=loss_dict["loss"]
        loss_per_dim.append(loss)

        # print("loss", loss)

    #normalize losses
    loss_normalization=max(loss_per_dim)
    weight_per_dim = [(x/loss_normalization).item() for x in loss_per_dim]



    #print them in order
    for i in range(64):
        # print("i", i, " w: ", weight_per_dim[i].item())
        print(weight_per_dim[i])

    with open("loss_weight_strand_latent.json", "w") as final:
        json.dump(weight_per_dim, final)

     
    #finished
    return


if __name__ == '__main__':
    main()
    
