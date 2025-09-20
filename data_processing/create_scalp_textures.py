#!/usr/bin/env python3

#creates all scalp textures containing latent for each strand


import sys
import os
import argparse
import torch
import torchvision
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models.strand_codec import StrandCodec
from torch.utils.data import DataLoader
from models.strand_codec import normalize_gt_data
from difflocks_utils.strand_util import compute_dirs
from difflocks_utils.vis_util import img_2_pca
import difflocks_utils.resize_right.resize_right as resize_right
import difflocks_utils.resize_right.interp_methods as interp_methods
from tqdm import tqdm
import numpy as np
from data_loader.dataloader import DiffLocksDataset
from data_loader.mesh_utils import world_to_tbn_space, closest_point_barycentrics, interpolate_tbn



#load some extensions
from torch.utils.cpp_extension import load
push_pull_inpaint = load(name="push_pull_inpaint_cpp", sources=["./extensions/push_pull_inpaint.cpp", "./extensions/cuda/push_pull_inpaint.cu"],
                         extra_include_paths=["./extensions/include"],
                        #  extra_cuda_cflags=["CUMAT_CONTEXT_USE_CUB_ALLOCATOR=0"]
                         )



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


def generate_scalp_textures(batch, model, normalization_dict, tex_size, output_scalp_texture_path):
    #gt batch
    gt_dict = prepare_gt_batch(batch)
    nr_strands = gt_dict["strand_positions"].shape[0]
    # print("nr strands", nr_strands)

    create_pca_pngs=False


    #get the values by encoding the strands
    gt_dict_normalized = normalize_gt_data(gt_dict, normalization_dict)
    #do it in chunks because otherwise we run out of memory
    nr_chunks=10
    gt_strands_pos = gt_dict_normalized["strand_positions"]
    gt_strands_dirs = gt_dict_normalized["strand_directions"]
    gt_strands_pos_chunked = torch.chunk(gt_strands_pos, nr_chunks)
    gt_strands_dirs_chunked = torch.chunk(gt_strands_dirs, nr_chunks)
    latents_chunked= []
    for i in range(len(gt_strands_pos_chunked)):
        cur_strands_pos = gt_strands_pos_chunked[i]
        cur_strands_dirs = gt_strands_dirs_chunked[i]
        gt_dict_chunk = {}
        gt_dict_chunk["strand_positions"]=cur_strands_pos
        gt_dict_chunk["strand_directions"]=cur_strands_dirs
        #run model and get latent for this chunk
        encoded_dict = model.encoder(gt_dict_chunk)
        latents=encoded_dict["z"]
        # print("latent shape",latents.shape)
        latents_chunked.append(latents)
    latents = torch.cat(latents_chunked)


    #assign directly to pixels
    strand_uv = batch["full_strands"]["root_uv"]
    strand_uv=strand_uv.cuda().squeeze(0)
    strand_uv_orig_flip = strand_uv
    strand_uv_orig_flip[:,1]=1.0-strand_uv_orig_flip[:,1]
    strand_indices = (strand_uv_orig_flip*tex_size).floor().int() #TODO Do we need to add a +0.5?
    scalp_texture = torch.zeros(1,64,tex_size,tex_size).cuda()
    scalp_texture[:,:,strand_indices[:,1],strand_indices[:,0]] = latents.transpose(0,1).unsqueeze(0) #make latents 1,64,nr_strands

    #get mask
    scalp_mask = torch.zeros(1,1,tex_size,tex_size).cuda()
    homogeneous_coord = torch.ones(nr_strands,1).cuda()
    scalp_mask[:,:,strand_indices[:,1],strand_indices[:,0]]=homogeneous_coord.transpose(0,1).unsqueeze(0)
    torchvision.utils.save_image(scalp_mask.squeeze(0), os.path.join(output_scalp_texture_path,"scalp_mask_"+str(tex_size)+".png"))
                                    

    #inpaint
    scalp_texture_inpainted=push_pull_inpaint.push_pull_inpaint(scalp_mask.squeeze(1), scalp_texture)
    if create_pca_pngs:
        scalp_texture_inpainted_pca = img_2_pca(scalp_texture_inpainted)
        torchvision.utils.save_image(scalp_texture_inpainted_pca.squeeze(0), os.path.join(output_scalp_texture_path,"scalp_texture_inpainted_pca_"+str(tex_size)+".png"))
    torch.save(scalp_texture_inpainted, os.path.join(output_scalp_texture_path, "scalp_texture_inpainted_"+str(tex_size)+".pt"))

    #make also downsized versions
    for i in range(4):
        scale_factor = 1.0/ (np.power(2,i+1))
        # print("scale_factor",scale_factor)
        scalp_texture_inpainted_downscaled = resize_right.resize(scalp_texture_inpainted, scale_factors=scale_factor, interp_method=interp_methods.linear)
        # print("scalp_texture_inpainted_downscaled",scalp_texture_inpainted_downscaled.shape)
        if create_pca_pngs:
            scalp_texture_inpainted_down_pca = img_2_pca(scalp_texture_inpainted_downscaled.contiguous())
            torchvision.utils.save_image(scalp_texture_inpainted_down_pca.squeeze(0), os.path.join(output_scalp_texture_path,"scalp_texture_inpainted_pca_"+str(int(tex_size*scale_factor))+".png"))
        out_path_downsampled_scalp=os.path.join(output_scalp_texture_path, "scalp_texture_inpainted_"+str(int(tex_size*scale_factor))+".pt")
        torch.save(scalp_texture_inpainted_downscaled, out_path_downsampled_scalp)

    #write a file to signify that we are done with this folder
    #start with x so that rsync copies it last if we copy to local 
    open( os.path.join(output_scalp_texture_path, "x_done.txt"), 'a').close()
        

def horizontally_flip(batch, scalp_mesh_data):

    # tbn=batch['full_strands']["tbn"].cuda()
    positions=batch['full_strands']["positions"][0].numpy()
    # print("positions", positions.shape)

    #flip
    positions[:,:,0]*=-1

    root_position = positions[:,0,:] #nr_strands x 3

    #it's more error prone to flip the tbn so we just recompute it
   
    mesh_verts=scalp_mesh_data["verts"]
    mesh_faces=scalp_mesh_data["faces"]
    mesh_v_tangents=scalp_mesh_data["v_tangents"]
    mesh_v_bitangents=scalp_mesh_data["v_bitangents"]
    mesh_v_normals=scalp_mesh_data["v_normals"]

    mesh_faces=mesh_faces.astype(np.int32)
    closest_points, barys, vertex_idxs, face_idxs=closest_point_barycentrics(root_position, mesh_verts, mesh_faces)

    # root_tangent, root_bitangent, root_normal = interpolate_tbn(self.root_position,  barys, vertex_idxs, mesh_v_tangents, mesh_v_bitangents, mesh_v_normals) 
    root_tangent, root_bitangent, root_normal = interpolate_tbn(barys, vertex_idxs, mesh_v_tangents, mesh_v_bitangents, mesh_v_normals) 
    #replace the normals because it's smoother
    root_normal=root_normal
    tbn = np.stack((root_tangent,root_bitangent,root_normal),axis=2) 

    #put it back on cuda
    tbn=torch.from_numpy(tbn).unsqueeze(0).cuda()
    positions=torch.from_numpy(positions).unsqueeze(0).cuda()
    root_normal=root_normal.unsqueeze(0).cuda()


    batch['full_strands']["tbn"]=tbn
    batch['full_strands']["positions"]=positions
    batch['full_strands']["root_normal"]=root_normal
    batch["full_strands"]["root_uv"][:,:,0] = 1 - batch["full_strands"]["root_uv"][:,:,0]


    return batch


def main():

    #argparse
    parser = argparse.ArgumentParser(description='Create scalp textures')
    parser.add_argument('--dataset_path', required=True, help='Path to the hair_synth dataset')
    parser.add_argument('--path_strand_vae_model', required=True, help='Path to .pt of the strandvae')
    parser.add_argument('--out_path', required=True, type=str, help='Where to output the processed hair_synth dataset')
    parser.add_argument('--skip_validity_check', dest='check_validity', action='store_false', help='Wether to check for the validity of each hairstyle we read from the dataset. Some older dataset versions might need this turned to false')
    args = parser.parse_args()

    
    tex_size=256

    print("args.check_validity",args.check_validity)

    difflocks_dataset = DiffLocksDataset(args.dataset_path, 
                                          check_validity=args.check_validity,
                                          load_full_strands=True,
                                          compute_tbn_full_strands=True,
                                        #   restrict_to_single_hairstyle_name="base_31_idx_9378" #for running local on output v5
                                            )
    loader = DataLoader(difflocks_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, persistent_workers=True)
    normalization_dict=difflocks_dataset.get_normalization_data()

    model = StrandCodec(do_vae=False, 
                        decode_type="dir",
                        scale_init=30.0,
                        nr_verts_per_strand=256, nr_values_to_decode=255,
                        dim_per_value_decoded=3).cuda()
    model.load_state_dict(torch.load(args.path_strand_vae_model))
    model = torch.compile(model)

    scalp_mesh, scalp_mesh_data = difflocks_dataset.get_scalp()

    progress_bar = tqdm(range(0, len(difflocks_dataset)), desc="Training progress")

    # for data in hair_synth_dataset:
    for batch in loader:
        progress_bar.update()

        #make the output path
        output_scalp_texture_path=os.path.join(args.out_path, "processed_hairstyles", batch["file"][0], "scalp_textures")
        os.makedirs(output_scalp_texture_path, exist_ok=True)
        if not os.path.isfile( os.path.join(output_scalp_texture_path,"x_done.txt")):
            #if it doesn't exist or we can't load it we create it
            generate_scalp_textures(batch, model, normalization_dict, tex_size, output_scalp_texture_path)

        
        #generate also a flipped texture, the reason being that just flipping the scalp texture does not result in a flipped hairstyle so we have to horizontally flip the data in the batch then encode a new flipped texture
        #make the output path
        output_scalp_texture_path=os.path.join(args.out_path, "processed_hairstyles", batch["file"][0], "scalp_textures_flip")
        os.makedirs(output_scalp_texture_path, exist_ok=True)
        if not os.path.isfile( os.path.join(output_scalp_texture_path,"x_done.txt")):
            batch=horizontally_flip(batch, scalp_mesh_data)
            #if it doesn't exist or we can't load it we create it
            generate_scalp_textures(batch, model, normalization_dict, tex_size, output_scalp_texture_path)
        



    #finished training
    return


if __name__ == '__main__':
    main()
    
