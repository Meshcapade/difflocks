import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import trimesh
import os
from pathlib import Path
import numpy as np
from torchvision.io import read_image
from os import listdir
from .mesh_utils import closest_point_barycentrics, compute_vertex_tbn, interpolate_tbn, compute_uv_space_data, mesh_to_data
import igl
import time
import json
import random
import torchvision
from tqdm.auto import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BODY_DATA_DIR = os.path.join(SCRIPT_DIR,"difflocks_bodydata")

def dir_empty(dir_path):
    return not any((True for _ in os.scandir(dir_path)))


class StrandsData(dict):
    def __init__(self, npz_data, meta_data, scalp_mesh_data, nr_full_strands_per_hairstyle, compute_tbn=True, flip=False):
        self.positions = npz_data["positions"] #nr_strands x nr_points_per_strand x 3
        self.root_normal = npz_data["root_normal"] #nr_strands x 3
        self.root_uv = npz_data["root_uv"] #nr_strands x 2
        self.root_position = npz_data["positions"][:,0,:] #nr_strands x 3
        print("init positiosis ", npz_data["positions"].shape)
        print("inti root position",self.root_position.shape)
       
        #solving a bug with the root_uvs, when the hair is mirrored by the blender pipeline the uv's were not mirrored also so we need to do it here
        if meta_data["mirror_hair"]:
            self.root_uv[:,0] = 1.0 - self.root_uv[:,0]

        if flip:
            self.positions[:,:,0]*=-1
            self.root_normal[:,0]*=-1
            self.root_uv[:,0] = 1 - self.root_uv[:,0]
            self.root_position[:,0]*=-1


        chunked_values = [100,1000]
        if nr_full_strands_per_hairstyle is not None and nr_full_strands_per_hairstyle not in chunked_values:
            self.subsample_to_nr_strands(nr_full_strands_per_hairstyle)

        if compute_tbn:
            self.compute_tbn(scalp_mesh_data["verts"], scalp_mesh_data["uv"],scalp_mesh_data["faces"],
                                    scalp_mesh_data["v_tangents"], scalp_mesh_data["v_bitangents"],scalp_mesh_data["v_normals"])


    def subsample_to_nr_strands(self, nr_strands_to_select):
        nr_strands_total = self.positions.shape[0] 
        indices_strands = np.random.randint(nr_strands_total, size=nr_strands_to_select)

        #new positions
        self.positions = self.positions[indices_strands,:,:]
        self.root_normal = self.root_normal[indices_strands,:]
        self.root_uv = self.root_uv[indices_strands,:]
        self.root_position = self.root_position[indices_strands,:]


    def compute_tbn(self, mesh_verts, mesh_uv, mesh_faces, mesh_v_tangents, mesh_v_bitangents, mesh_v_normals):
        mesh_faces=mesh_faces.astype(np.int32)
        closest_points, barys, vertex_idxs, face_idxs=closest_point_barycentrics(self.root_position, mesh_verts, mesh_faces)

        root_tangent, root_bitangent, root_normal = interpolate_tbn(barys, vertex_idxs, mesh_v_tangents, mesh_v_bitangents, mesh_v_normals) 
        #replace the normals because it's smoother
        self.root_normal=root_normal
        tbn = np.stack((root_tangent,root_bitangent,root_normal),axis=2) 
        self.tbn=tbn
        # print("tbn", tbn.shape)



    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            d[key]=value
        return d
        

 



class HairStylePathData():
    def __init__(self, hairstyle_path):
        self.hairstyle_path=hairstyle_path

        #check the chunked versions
        #for full_strands
        # path_chunk_10 = os.path.join(hairstyle_path,"full_strands_chunked","nr_strands_10")
        self.path_chunk_100 = os.path.join(hairstyle_path,"full_strands_chunked","nr_strands_100")
        self.path_chunk_1000 = os.path.join(hairstyle_path,"full_strands_chunked","nr_strands_1000")
        # self.chunked_10_strands_files = [os.path.join(path_chunk_10, f) for f in listdir(path_chunk_10)]
        # self.chunked_100_strands_files = [os.path.join(path_chunk_100, f) for f in listdir(path_chunk_100)]
        # self.chunked_1000_strands_files = [os.path.join(path_chunk_1000, f) for f in listdir(path_chunk_1000)]

    # check if we even have the chunked data required
    def can_load_chunked_paths(self,chunk=100):
        if chunk==100:
            return os.path.isdir(self.path_chunk_100) and not dir_empty(self.path_chunk_100)
        elif chunk==1000:
            return os.path.isdir(self.path_chunk_1000) and not dir_empty(self.path_chunk_1000)
        else:
            return False

    # to save memory and process time, we load the chunked paths of the granularity that we expect to read from later
    def load_chunked_paths(self,chunk=100):
        if chunk==100:
            self.chunked_100_strands_files = [os.path.join(self.path_chunk_100, f) for f in listdir(self.path_chunk_100)]
        elif chunk==1000:
            self.chunked_1000_strands_files = [os.path.join(self.path_chunk_1000, f) for f in listdir(self.path_chunk_1000)]
        else:
            print('please check chunk value')
            exit()

       

class DiffLocksDataset(Dataset):
    def __init__(self, 
                difflocks_path, 
                processed_difflocks_path=None, #contains path to scalp textures
                train=None, 
                load_rgb_imgs=False,
                load_hair_mask=False,
                load_orientation_map=False,
                load_density_imgs=False,
                load_full_strands=False, 
                load_guide_strands=False,
                load_interpolated_strands=False, 
                load_cam=False,
                load_path=True,
                load_material=False,
                load_scalp_texture=False,
                scalp_texture_resolution = 256,
                load_latents=False,
                latents_type_list=["sdxl"],
                load_latents_layers=[],
                compute_tbn_full_strands=False,
                compute_tbn_guide_strands=False,
                compute_tbn_interpolated_strands=False,
                # compute_fourier_full_strands=False,
                nr_full_strands_per_hairstyle=None,
                subsample_factor=None,
                check_validity=True,
                check_validity_processed=True,
                do_pedantic_checks=True, #some checks may be redundant when doing on lambda where we know that numpy_state is the last thing to write, but may be worth doing locally where the data is not fully downloaded
                overfit=False,
                restrict_to_single_hairstyle_name=None,
                train_ratio = 0.9,
                randomly_flip=False,
                max_nr_samples=None
                ):
        
        if subsample_factor is not None:
            assert 768% (1.0/subsample_factor) == 0 #The subsample factor does not cleanly divide the image size

        self.difflocks_path=difflocks_path
        self.processed_difflocks_path=processed_difflocks_path
        self.train=train 
        self.load_rgb_imgs=load_rgb_imgs
        self.load_hair_mask=load_hair_mask
        self.load_orientation_map=load_orientation_map
        self.load_density_imgs= load_density_imgs
        self.load_full_strands=load_full_strands
        self.load_guide_strands=load_guide_strands
        self.load_interpolated_strands=load_interpolated_strands
        self.load_cam=load_cam
        self.load_path=load_path
        self.load_material=load_material
        self.load_scalp_texture = load_scalp_texture
        self.scalp_texture_resolution=scalp_texture_resolution
        self.load_latents=load_latents
        self.load_latents_layers= load_latents_layers
        self.latents_type_list=latents_type_list
        self.compute_tbn_full_strands=compute_tbn_full_strands
        self.compute_tbn_guide_strands=compute_tbn_guide_strands
        self.compute_tbn_interpolated_strands=compute_tbn_interpolated_strands
        # self.compute_fourier_full_strands=compute_fourier_full_strands
        #each hairstyle may have different number of strands which makes it kinda annoying to batch together. We could for example just get a random subsample of strands from each hairstyle
        self.nr_full_strands_per_hairstyle=nr_full_strands_per_hairstyle
        self.subsample_factor=subsample_factor
        self.overfit=overfit
        self.randomly_flip = randomly_flip #we need to do the horizontal flipping in the dataloader since flipping a hairstyle doesn't just mean flippin the scalp texture but rather we compute a whole other scalp texture from flipped strands, we here in the dataloader we need to load another file for the scalp texture


        #get the paths for all the generated hairstyles
        print("difflocks loader reading paths to all hairstyles")
        hairstyles_path_list= [os.path.join(difflocks_path,"generated_hairstyles",v) for v in os.listdir(os.path.join(difflocks_path,"generated_hairstyles"))]
        # print("hairstyles_path_list", hairstyles_path_list)

        if max_nr_samples is not None:
            hairstyles_path_list=hairstyles_path_list[0:max_nr_samples]


        #for each hairstyle check if all the expected files are there, if any are missing we skip that particular hairstyle
        print("filtering hairstyle paths")
        hairstyles_path_list_filtered= []
        for hairstyle_path in tqdm(hairstyles_path_list):
            basename = os.path.basename(hairstyle_path) #will be something like base_X_idx_Y


            is_valid=True
            if check_validity: #sometimes we just want to skip the validity check because the dataset may only be partially downloaded
                #the last thing that is written to the folder, so the frist thing we check for and if it doesn't exist we don't bother to check the rest
                if not os.path.isfile(os.path.join(hairstyle_path,"numpy_state.npz")):
                    continue

                if do_pedantic_checks:
                    if not os.path.isfile(os.path.join(hairstyle_path,"metadata.json")):
                        # is_valid=False
                        continue
                    if load_rgb_imgs and not os.path.isfile(os.path.join(hairstyle_path,"rgb.png")):
                        # is_valid=False
                        continue
                    if load_full_strands and nr_full_strands_per_hairstyle==None and not os.path.isfile(os.path.join(hairstyle_path,"full_strands.npz")):
                        # is_valid=False
                        continue
                    if load_guide_strands and not os.path.isfile(os.path.join(hairstyle_path,"guide_strands.npz")):
                        # is_valid=False
                        continue
                    if load_interpolated_strands and not os.path.isfile(os.path.join(hairstyle_path,"interpolated_strands.npz")):
                        # is_valid=False
                        continue
                    if load_cam and not os.path.isfile(os.path.join(hairstyle_path,"cam.npz")):
                        # is_valid=False
                        continue
                    #if we are loading any chunked data but those chunks are empty
                    if nr_full_strands_per_hairstyle is not None:
                        if self.nr_full_strands_per_hairstyle<=100:
                            path_chunk_100 = os.path.join(hairstyle_path,"full_strands_chunked","nr_strands_100")
                            is_invalid = not os.path.isdir(path_chunk_100) or dir_empty(path_chunk_100)
                            if is_invalid:
                                # is_valid=False
                                continue
                        elif self.nr_full_strands_per_hairstyle>100 and self.nr_full_strands_per_hairstyle<=1000:
                            path_chunk_1000 = os.path.join(hairstyle_path,"full_strands_chunked","nr_strands_1000")
                            is_invalid = not os.path.isdir(path_chunk_1000) or dir_empty(path_chunk_100)
                            if is_invalid:
                                # is_valid=False
                                continue

                    #probably the only validity check we actually need since this file is always written at the end
                    # if "output_v5" not in difflocks_path: #output_v5 doesn't have this data sdo we can't check for it either way
                    npz_path = os.path.join(hairstyle_path,"numpy_state.npz")
                    # if not os.path.isfile(npz_path):
                        # is_valid=False
                    # try to load it to make sure it works
                    try:
                        data = np.load(npz_path, allow_pickle=True)
                    except:
                        print("couldn't load numpy_state for", hairstyle_path)
                        is_valid=False


                #try to load also metadata because for some reason some of them fail at this
                #we already blacklisted the hairstyle with the bad metadata
                # if os.path.isfile(os.path.join(hairstyle_path,"metadata.json")):
                #     with open(os.path.join(hairstyle_path,"metadata.json"),'r') as f:
                #         try:
                #             meta_data = f.read()
                #             meta_data = json.loads(meta_data)
                #             if isinstance(meta_data, str):
                #                 print("why is this a str", hairstyle_path)
                #                 is_valid=False
                #         except:
                #             print("couldn't load metadata for", hairstyle_path)
                #             is_valid=False



            if processed_difflocks_path is not None and check_validity_processed:
                #we check if we have the scalp_texture flipped at 16 resolution since that is the last one that is written
                if load_scalp_texture:
                    #check the x_done was written
                    scalp_texture_done_path = os.path.join(processed_difflocks_path, "processed_hairstyles", basename, "scalp_textures", "x_done.txt")
                    scalp_texture_done_path_flip = os.path.join(processed_difflocks_path, "processed_hairstyles", basename, "scalp_textures_flip", "x_done.txt")
                    if not os.path.isfile(scalp_texture_done_path):
                        # is_valid=False
                        continue
                    if not os.path.isfile(scalp_texture_done_path_flip):
                        # is_valid=False
                        continue
                    
                    
                if load_latents or bool(load_latents_layers):
                    for latents_type in latents_type_list:
                        subsample_factor_to_check_latents_at=1 #most latents are made at subsample factor 1 and internally the images are resampled to whatever necessary. For example CLIP latents loads images at full resolution but before the preprocessor we resize to 336x336
                        if latents_type=="sdxl":
                            subsample_factor_to_check_latents_at=subsample_factor
                            
                        #check if X_done for the latents for this specific type and for this subsample factor actually exists
                        latents_done_path = os.path.join(processed_difflocks_path, "processed_hairstyles", basename, "latents_"+latents_type+"_subsample_"+str(subsample_factor_to_check_latents_at), "x_done.txt")
                        latents_done_path_flip = os.path.join(processed_difflocks_path, "processed_hairstyles", basename, "latents_flipped_"+latents_type+"_subsample_"+str(subsample_factor_to_check_latents_at), "x_done.txt")
                        if not os.path.isfile(latents_done_path):
                            # is_valid=False
                            continue
                        if not os.path.isfile(latents_done_path_flip):
                            # is_valid=False
                            continue
                    


            #HACK for output_v5 to load when we use the gloss extensions
            # path_chunk_100=os.path.join(hairstyle_path,"full_strands_chunked","nr_strands_100")
            # if not os.path.isdir(path_chunk_100):
            #     is_valid=False
            # #check if ther are any files there
            # if os.path.isdir(path_chunk_100):
            #     if len(listdir(path_chunk_100))<2:
            #         is_valid=False
                    

            #some of the generated hairstyles have issues so we just ignore them
            blacklist=["base_69_idx_6757","base_60_idx_7154","base_42_idx_8758"]
            if basename in blacklist:
                is_valid=False

            
            #if we are restricting to only one hairstyle, only add the one we selected
            if restrict_to_single_hairstyle_name is not None and basename!=restrict_to_single_hairstyle_name:
                is_valid=False

                

            if is_valid:
                hairstyles_path_list_filtered.append(hairstyle_path)

        

        hairstyles_path_list=hairstyles_path_list_filtered

        print("hairstyles_path_list after filtering", len(hairstyles_path_list)) 

        #depending if we are using train or test dataset we get a different subset of all the generated hairstyles
        # print("total hairstyles_path_list",hairstyles_path_list)
        nr_hairstyles= len(hairstyles_path_list)
        if train is not None:
            if train:
                hairstyles_path_list=hairstyles_path_list[:int(nr_hairstyles*train_ratio)]
            else:
                hairstyles_path_list=hairstyles_path_list[int(nr_hairstyles*train_ratio):]

        print("hairstyles_path_list after filtering for train/test", len(hairstyles_path_list)) 
        
        # print("----------")
        # print("selected hairstyles_path_list",hairstyles_path_list)
        # exit(1)

        self.hairstyles_path_list=hairstyles_path_list
        # self.hairstyles_path_data_list = [HairStylePathData(v) for v in self.hairstyles_path_list]
        self.hairstyles_path_data_list = []
        for v in self.hairstyles_path_list:
            hairstyle_pathdata = HairStylePathData(v)
            #if we expect to read some chunked data, load also the paths to the chunked npzs
            if self.nr_full_strands_per_hairstyle is not None:
                if self.nr_full_strands_per_hairstyle<=100:
                    # if not hairstyle_pathdata.can_load_chunked_paths(100):
                        # continue
                    hairstyle_pathdata.load_chunked_paths(100)
                elif self.nr_full_strands_per_hairstyle>100 and self.nr_full_strands_per_hairstyle<=1000:
                    # if not hairstyle_pathdata.can_load_chunked_paths(1000):
                        # continue
                    hairstyle_pathdata.load_chunked_paths(1000)
            self.hairstyles_path_data_list.append(hairstyle_pathdata)


        self.body_data_path=os.path.join(difflocks_path,"body_data")

        #load scalp
        self.scalp_mesh, self.scalp_mesh_data= self.compute_scalp_data(os.path.join(self.body_data_path,"scalp.ply"))
        



        self.smplx_base_mesh = trimesh.load(os.path.join(self.body_data_path,"smplx_base.ply"))
        self.smplx_base_mesh_data={}
        smplx_base_v, smplx_base_f, smplx_base_uv = mesh_to_data(self.smplx_base_mesh) 
        self.smplx_base_mesh_data["verts"]=smplx_base_v
        self.smplx_base_mesh_data["faces"]=smplx_base_f
        self.smplx_base_mesh_data["uv"]=smplx_base_uv


        #TODO load also the flame head

        print("Finished initializing DiffLocksDataset")


    def __len__(self):
        # print("self.hairstyles_path_list",self.hairstyles_path_list)
        return len(self.hairstyles_path_list)

    def __getitem__(self, idx):
        out_dict = {}

        if self.overfit==True:
            # idx=4
            idx=5
            np.random.seed(0)

        hairstyle_data= self.hairstyles_path_data_list[idx]
        hairstyle_path = hairstyle_data.hairstyle_path

        #if we need any of the processed data, we load also the path for it
        # if self.load_scalp_texture:
        if self.processed_difflocks_path is not None:
            hairstyle_path_processed = os.path.join(self.processed_difflocks_path, "processed_hairstyles", os.path.basename(hairstyle_path))

        # print("hairstyle_path",hairstyle_path)
            

        flip = False
        if self.randomly_flip:
            if random.random()<0.5:
                flip = True

            
        #read metadata
        with open(os.path.join(hairstyle_path,"metadata.json"),'r') as f:
            try:
                meta_data = f.read()
                meta_data = json.loads(meta_data)
                if isinstance(meta_data, str):
                    print("why is this a str", hairstyle_path)
                    exit(1)
            except:
                print("couldn't load metadata for", hairstyle_path)
                exit(1)


        if self.load_rgb_imgs:
            img_path = os.path.join(hairstyle_path,"rgb.png")
            img = read_image(img_path).float()/255.0
            if flip:
                img=torchvision.transforms.functional.hflip(img)
            if self.subsample_factor is not None:
                scale_factor=[1.0/self.subsample_factor, 1.0/self.subsample_factor]
                img=torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode="area").squeeze(0)
            out_dict["rgb_img"] = img

        if self.load_hair_mask:
            img_path = os.path.join(hairstyle_path_processed,"hair_masks","hair_mask.png")
            if flip:
                img_path = os.path.join(hairstyle_path_processed,"hair_masks_flip","hair_mask.png")
            img = read_image(img_path).float()/255.0
            img = img[0:1,:,:]
            if self.subsample_factor is not None:
                scale_factor=[1.0/self.subsample_factor, 1.0/self.subsample_factor]
                img=torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode="nearest").squeeze(0)
            out_dict["hair_mask"] = img

        if self.load_orientation_map:
            img_path = os.path.join(hairstyle_path_processed,"orientation_maps","orientation_map.png")
            if flip:
                img_path = os.path.join(hairstyle_path_processed,"orientation_maps_flip","orientation_map.png")
            img = read_image(img_path).float()/255.0
            img = img[0:1,:,:]
            if self.subsample_factor is not None:
                scale_factor=[1.0/self.subsample_factor, 1.0/self.subsample_factor]
                img=torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode="nearest").squeeze(0)
            out_dict["orientation_map"] = img

        if self.load_density_imgs:
            img_path = os.path.join(hairstyle_path,"density.png")
            img = read_image(img_path)[0:1,:,:].float()/255.0
            if flip:
                img=torchvision.transforms.functional.hflip(img)
            out_dict["density_img"] = img

        if self.load_full_strands:
            #if we load a particular number of strands, figure out what is the most efficient npz file to load 
            if self.nr_full_strands_per_hairstyle:
                # if self.nr_full_strands_per_hairstyle<=10:
                    # npz_path = np.random.choice(hairstyle_data.chunked_10_strands_files)
                # if self.nr_full_strands_per_hairstyle>10 and self.nr_full_strands_per_hairstyle<=100:
                if self.nr_full_strands_per_hairstyle<=100:
                    npz_path = np.random.choice(hairstyle_data.chunked_100_strands_files)
                elif self.nr_full_strands_per_hairstyle>100 and self.nr_full_strands_per_hairstyle<=1000:
                    npz_path = np.random.choice(hairstyle_data.chunked_1000_strands_files)
                elif self.nr_full_strands_per_hairstyle>1000 and self.nr_full_strands_per_hairstyle<=10000:
                    npz_path = os.path.join(hairstyle_path,"full_strands_10000.npz")
                else:
                    npz_path = os.path.join(hairstyle_path,"full_strands.npz")
            else:
                # npz_name = "full_strands_100.npz"
                npz_path = os.path.join(hairstyle_path,"full_strands.npz")

            # npz_path = os.path.join(hairstyle_path,"full_strands.npz")

                
            # print("loading", npz_path)
                

            
            strand_data = StrandsData(np.load(npz_path), meta_data, self.scalp_mesh_data, self.nr_full_strands_per_hairstyle, self.compute_tbn_full_strands, flip)
            out_dict["full_strands"] = strand_data.to_dict()

        if self.load_guide_strands:
            guide_strands_path = os.path.join(hairstyle_path, "guide_strands.npz")
            strand_data = StrandsData(np.load(guide_strands_path), meta_data, self.scalp_mesh_data, None, self.compute_tbn_guide_strands, flip)
            out_dict["guide_strands"] = strand_data.to_dict()


        if self.load_interpolated_strands:
            interpolated_strands_path = os.path.join(hairstyle_path, "interpolated_strands.npz")
            strand_data = StrandsData(np.load(interpolated_strands_path), meta_data, self.scalp_mesh_data, None, self.compute_tbn_interpolated_strands, flip)
            out_dict["interpolated_strands"] = strand_data.to_dict()

            

        #load cam
        if self.load_cam:
            data = np.load(os.path.join(hairstyle_path,"cam.npz"))
            data_dict = dict((k, v) for k, v in data.items())

            #we need to also flip the camera x position
            if flip:
                # raise RuntimeError("We cannot load camera and do random flips because we haven't yet implemented the flip for the camera location")
                flip_matrix = np.array([[-1.0, 0.0,  0.0, 0.0],
                        [ 0.0, 1.0,  0.0, 0.0],
                        [ 0.0, 0.0,  1.0, 0.0],
                        [ 0.0, 0.0,  0.0, 1.0]])
                RT = data_dict["RT"]
                RT_world_cam = np.linalg.inv(RT)
                RT_world_cam = np.dot(flip_matrix, RT_world_cam)
                RT = np.linalg.inv(RT_world_cam)
                data_dict["RT"]=RT

            #get lookat dir
            tf_cam_world=data_dict["RT"].astype(np.float32)
            tf_world_cam=np.linalg.inv(data_dict["RT"]).astype(np.float32)
            lookdir=tf_world_cam[0:3,2].reshape((3))
            lookdir = lookdir / np.linalg.norm(lookdir)


            #get the dict into one that contains both tf_cam_world and tf_world_cam
            data_dict_v2={}
            data_dict_v2["tf_cam_world"]=tf_cam_world
            data_dict_v2["tf_world_cam"]=tf_world_cam
            data_dict_v2["K"]=data_dict["K"].astype(np.float32)
            data_dict_v2["pos_in_world"]=data_dict_v2["tf_world_cam"][0:3,3].reshape((3)).astype(np.float32)
            data_dict_v2["height"]=768
            data_dict_v2["width"]=768
            data_dict_v2["lookdir"]=lookdir

            #resizing if necessary
            if self.subsample_factor is not None:
                data_dict_v2["K"][0:2,:] = data_dict_v2["K"][0:2,:]/self.subsample_factor
                data_dict_v2["height"]=int(data_dict_v2["height"]/self.subsample_factor)
                data_dict_v2["width"]=int(data_dict_v2["width"]/self.subsample_factor)


            out_dict["cam"] = data_dict_v2

        if self.load_path:
            out_dict["path"]=hairstyle_path
            out_dict['file']=os.path.basename(hairstyle_path)

        if self.load_material:
            materials_vals=[]
            materials_vals.append( meta_data["material_wave_scale"] )
            materials_vals.append( meta_data["material_wave_phase_offset"] )
            materials_vals.append( meta_data["material_wave_strength"] )
            materials_vals.append( meta_data["material_melanin_amount"] )
            materials_vals.append( meta_data["bsdf_melanin_redness"] )
            materials_vals.append( meta_data["bsdf_roughness"] )
            materials_vals.append( meta_data["bsdf_radial_roughness"] )
            materials_vals.append( meta_data["bsdf_coat"] )
            materials_vals.append( meta_data["root_darkness_start"] )
            materials_vals.append( meta_data["root_darkness_end"] )
            materials_vals.append( meta_data["root_darkness_strength"] )
            out_dict["material"]= np.array(materials_vals).astype(np.float32)
         


        if self.load_scalp_texture:
            if flip:
                scalp_texture_path = os.path.join(hairstyle_path_processed, "scalp_textures_flip", "scalp_texture_inpainted_"+str(self.scalp_texture_resolution)+".pt")
            else:
                scalp_texture_path = os.path.join(hairstyle_path_processed, "scalp_textures", "scalp_texture_inpainted_"+str(self.scalp_texture_resolution)+".pt")
            scalp_texture = torch.load(scalp_texture_path, torch.device('cpu'), weights_only=True).squeeze(0)
            # out_dict["scalp_texture"] = scalp_texture["scalp_texture"].transpose(2,0,1)
            out_dict["scalp_texture"] = scalp_texture
            # out_dict["binary_map"] = scalp_texture["binary_map"].transpose(2,0,1)

        #select the resolution of latent we will be loading and setup the dicts
        if self.load_latents or bool(self.load_latents_layers):
            out_dict["latents"]={}
            for l_idx, latents_type in enumerate(self.latents_type_list):
                out_dict["latents"][latents_type]={}

                #which layers ot load for this latent type
                layer_types=self.load_latents_layers[l_idx]

                #select the resolution to load for this latent
                subsample_factor_latents=1 #most latents are made at subsample factor 1 and internally the images are resampled to whatever necessary. For example CLIP latents loads images at full resolution but before the preprocessor we resize to 336x336
                if latents_type=="sdxl" or latents_type=="dinov2": #sdxl loads images at a specific resolution
                    subsample_factor_latents=self.subsample_factor

                #load final latent
                # if self.load_latents:
                if "final_latent" in layer_types:
                    if flip:
                        latent_path = os.path.join(hairstyle_path_processed, "latents_flipped_"+latents_type+"_subsample_"+str(subsample_factor_latents), "final_latent.pt") 
                    else:
                        latent_path = os.path.join(hairstyle_path_processed, "latents_"+latents_type+"_subsample_"+str(subsample_factor_latents), "final_latent.pt") 
                    final_latent = torch.load(latent_path, torch.device('cpu'), weights_only=True).squeeze(0)
                    out_dict["latents"][latents_type]["final_latent"] = final_latent

                #load cls token
                if "cls_token" in layer_types:
                    if flip:
                        latent_path = os.path.join(hairstyle_path_processed, "latents_flipped_"+latents_type+"_subsample_"+str(subsample_factor_latents), "cls_token.pt") 
                    else:
                        latent_path = os.path.join(hairstyle_path_processed, "latents_"+latents_type+"_subsample_"+str(subsample_factor_latents), "cls_token.pt") 
                    cls_token = torch.load(latent_path, torch.device('cpu'), weights_only=True).squeeze(0)
                    out_dict["latents"][latents_type]["cls_token"] = cls_token

                #load multires one
                # if bool(self.load_latents_layers):
                #check if it there is any layer_x
                for layer_name in layer_types:
                    if "layer_" in layer_name:
                        layer_nr=int(layer_name.replace('layer_', ''))
                        if flip:
                            latents_path = os.path.join(hairstyle_path_processed, "latents_flipped_"+latents_type+"_subsample_"+str(subsample_factor_latents)) 
                        else:
                            latents_path = os.path.join(hairstyle_path_processed, "latents_"+latents_type+"_subsample_"+str(subsample_factor_latents)) 
                        #layer x
                        latent_path=os.path.join(latents_path,"latent_layer_"+str(layer_nr)+".pt")
                        latent = torch.load(latent_path, torch.device('cpu'), weights_only=True).squeeze(0)
                        out_dict["latents"][latents_type]["layer_"+str(layer_nr)] = latent

      
           


           
        return out_dict

    
    def get_scalp(self):
        return self.scalp_mesh, self.scalp_mesh_data
    
    def get_base_mesh(self):
        return self.smplx_base_mesh, self.smplx_base_mesh_data
    
    @staticmethod
    def get_normalization_data():
        normalization_dict={}

        ###values computed with compute_mean_and_std.py
        #for data whitening
        #center the strand data to be drawn from unit gaussian

        #For strand data contining xyz positions
        xyz_mean = torch.tensor([[-0.0001, -0.0080, -0.0602]]).cuda()
        xyz_std = torch.tensor([0.0600, 0.0564, 0.0562]).cuda()
        normalization_dict["xyz_mean"]=xyz_mean
        normalization_dict["xyz_std"]=xyz_std

        dir_mean = torch.tensor([[-8.7517e-07, -9.2789e-05, -4.3253e-04]]).cuda()
        dir_std = torch.tensor([0.0005, 0.0005, 0.0004]).cuda()
        normalization_dict["dir_mean"]=dir_mean
        normalization_dict["dir_std"]=dir_std

        curv_mean = torch.tensor([[ 6.2361e-09, -2.4707e-06,  3.8570e-07]]).cuda()
        curv_std = torch.tensor([3.0512e-05, 3.7916e-05, 2.7756e-05]).cuda()
        normalization_dict["curv_mean"]=curv_mean
        normalization_dict["curv_std"]=curv_std

        return normalization_dict

    @staticmethod
    def compute_scalp_data(scalp_path):
        scalp_mesh = trimesh.load(scalp_path)
        scalp_mesh_data={}
        scalp_v, scalp_f, scalp_uv = mesh_to_data(scalp_mesh)
        scalp_v_tangents, scalp_v_bitangents, scalp_v_normals = compute_vertex_tbn(scalp_v, scalp_uv, scalp_f.astype(np.int32))
        scalp_mesh_data["verts"]=scalp_v
        scalp_mesh_data["faces"]=scalp_f
        scalp_mesh_data["uv"]=scalp_uv
        scalp_mesh_data["v_tangents"]=scalp_v_tangents
        scalp_mesh_data["v_bitangents"]=scalp_v_bitangents
        scalp_mesh_data["v_normals"]=scalp_v_normals
        # #have it also on gpu to avoid too many transfers between cpu and gpu
        # self.scalp_mesh_data_gpu={}
        # self.scalp_mesh_data_gpu["verts"]=torch.from_numpy(scalp_v).cuda()
        # self.scalp_mesh_data_gpu["faces"]=torch.from_numpy(scalp_f).cuda()
        # self.scalp_mesh_data_gpu["uv"]=torch.from_numpy(scalp_uv).cuda()
        #get also tbn for each vertex of the mesh

        #compute also a rasterized map of barycentric coordinates and face indices
        scalp_uv_maps_size=512
        scalp_index_map, scalp_vertex_idxs_map, scalp_bary_map= compute_uv_space_data(torch.from_numpy(scalp_uv), torch.from_numpy(scalp_f.astype(np.int32)), scalp_uv_maps_size)
        scalp_mesh_data["index_map"]=scalp_index_map
        scalp_mesh_data["vertex_idxs_map"]=scalp_vertex_idxs_map
        scalp_mesh_data["bary_map"]=scalp_bary_map

        return scalp_mesh, scalp_mesh_data
    




        