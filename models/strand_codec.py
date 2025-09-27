import os

import torch
import torch.nn as nn

from modules.difflocks.difflocks_modules.networks import LinearDummy, LinearWN_v2, Conv1dWN_v2, BlockSiren, kaiming_init
from modules.difflocks.difflocks_utils.general_util import batched_index_select
import json


def normalize_data_3D(strands_data, mean, std):
    # print("strands_data ",strands_data.shape)
    orig_shape = strands_data.shape
    nr_strands = strands_data.shape[0]
    nr_elements_spatial_dim = strands_data.shape[1]
    strands_data=strands_data.view(nr_strands, nr_elements_spatial_dim, -1)
    # print("strands_data ",strands_data.shape)
    # print("mean ",mean.shape)
    strands_data=strands_data-mean.view(1,1,-1)
    strands_data=strands_data/(std.view(1,1,-1)+1e-6)

    #checks which elements are zero in the std and just set them to zero because they are basically noise
    valid=std.view(1,1,-1)>=1e-9
    strands_data=strands_data*valid

    strands_data=strands_data.view(orig_shape)
    return strands_data

def normalize_data_2D(strands_data, mean, std):
    # print("strands_data ",strands_data.shape)
    orig_shape = strands_data.shape
    nr_strands = strands_data.shape[0]
    strands_data=strands_data.view(nr_strands, -1)
    # print("strands_data ",strands_data.shape)
    # print("mean ",mean.shape)
    strands_data=strands_data-mean.view(1,-1)

    strands_data=strands_data/(std.view(1,-1)+1e-6)

    #checks which elements are zero in the std and just set them to zero because they are basically noise
    valid=std.view(1,-1)>=1e-9
    strands_data=strands_data*valid

    strands_data=strands_data.view(orig_shape)
    return strands_data

def un_normalize_data(strands_data, mean, std):
    orig_shape = strands_data.shape
    nr_strands = strands_data.shape[0]
    nr_elements_spatial_dim = strands_data.shape[1]
    strands_data=strands_data.view(nr_strands, nr_elements_spatial_dim, -1)
    strands_data=strands_data*std.view(1,1,-1)
    strands_data=strands_data+mean.view(1,1,-1)
    strands_data=strands_data.view(orig_shape)
    return strands_data

def un_normalize_data_2D(strands_data, mean, std):
    orig_shape = strands_data.shape
    nr_strands = strands_data.shape[0]
    strands_data=strands_data.view(nr_strands, -1)
    strands_data=strands_data*std.view(1,-1)
    strands_data=strands_data+mean.view(1,-1)
    strands_data=strands_data.view(orig_shape)
    return strands_data

def normalize_gt_data(gt_dict, normalization_dict):
    # mean=normalization_dict["xyz_mean"]
    # std=normalization_dict["xyz_std"]
    # strands_data=normalize_data(strands_data, mean, std)

    gt_dict_out = {}

    gt_strand_positions=normalize_data_3D(gt_dict["strand_positions"], normalization_dict["xyz_mean"], normalization_dict["xyz_std"])
    gt_dict_out["strand_positions"]=gt_strand_positions

    gt_strand_dirs=normalize_data_3D(gt_dict["strand_directions"], normalization_dict["dir_mean"], normalization_dict["dir_std"])
    gt_dict_out["strand_directions"]=gt_strand_dirs

    # gt_strand_curv=normalize_data_3D(gt_dict["strand_curvatures"], normalization_dict["curv_mean"], normalization_dict["curv_std"])
    # gt_dict_out["strand_curvatures"]=gt_strand_curv

    if "stft_directions" in gt_dict:
        gt_stft_dirs=normalize_data_3D(gt_dict["stft_directions"], 
                                    normalization_dict["stft_64_32_64_mean_across_freq"], 
                                    normalization_dict["stft_64_32_64_std_across_freq"])
        gt_dict_out["stft_directions"]=gt_stft_dirs

    if "fft_directions" in gt_dict:
        print("before normalization",gt_dict["fft_directions"].mean())
        gt_fft_dirs=normalize_data_2D(gt_dict["fft_directions"], 
                                    normalization_dict["fft_mean"], 
                                    normalization_dict["fft_std"])

        # gt_fft_dirs=gt_dict["fft_directions"]
        print("after normalization",gt_fft_dirs.mean())
        gt_dict_out["fft_directions"]=gt_fft_dirs

    # gt_dict_out["fft_directions"]=gt_dict["fft_directions"]

    return gt_dict_out

def align_pred_with_gt(pred_dict, normalization_dict):
    #the positions (originating from integration of directions) from the codec start at 0 but we need to shift this by the amount that normalize_gt_data shifts the origin
    strands_data=pred_dict["strand_positions"]
    strands_data=strands_data-normalization_dict["xyz_mean"].view(1,1,3)/normalization_dict["xyz_std"].view(1,1,3)
    pred_dict_out = dict(pred_dict)
    pred_dict_out["strand_positions"]=strands_data
    return pred_dict_out



class StrandEncoder1dCNNWN(nn.Module):
    def __init__(self, do_vae, out_channels=128, num_pts=256):
        super(StrandEncoder1dCNNWN, self).__init__()

        self.do_vae = do_vae
        self.num_pts = num_pts

        # self.training = False


        in_channels = 0
        in_channels += 3 # 3 for xyz
        in_channels += 3 # 3 for dirs

        
        self.cnn_encoder = torch.nn.Sequential(
            Conv1dWN_v2(in_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode='replicate'),torch.nn.SiLU(),
            Conv1dWN_v2(64, 64, kernel_size=4, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
            Conv1dWN_v2(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
            Conv1dWN_v2(128, 128, kernel_size=4, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
            Conv1dWN_v2(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
            Conv1dWN_v2(256, 256, kernel_size=4, stride=2, padding=1, padding_mode='replicate'), torch.nn.SiLU(),
        )


        self.aggregate_towards_mean = torch.nn.Sequential(
            LinearWN_v2(256 * 4, 512), torch.nn.SiLU(),
        )
        self.pred_mean = LinearWN_v2(512, out_channels)

        self.aggregate_towards_logstd = torch.nn.Sequential(
            LinearWN_v2(256 * 4, 512), torch.nn.SiLU(),
        )
        self.pred_logstd = LinearWN_v2(512, out_channels)
       

        self.apply(lambda x: kaiming_init(x, False, nonlinearity="silu"))
        kaiming_init(self.pred_mean, True)
        kaiming_init(self.pred_logstd, True)


        self.tanh = torch.nn.Tanh()


    def forward(self, gt_dict):

        points=gt_dict["strand_positions"]
        dirs=gt_dict["strand_directions"]

        #points
        points = points.permute(0, 2, 1) ## nr_strands, xyz, 100
        nr_strands = points.shape[0]
        #dirs
        last_dir = dirs[:, -1:, :]
        dirs = torch.cat([dirs, last_dir],1) # make the direction nr_strands, 100, 3
        dirs = dirs.permute(0, 2, 1)

        per_point_features = torch.cat([points, dirs] ,1)
        # per_point_features = points
        x=per_point_features

        # print("x",x.mean(),x.std())

        strand_features = self.cnn_encoder(x) # nr_strands, 128(nr_features), 3(elements per string)

        # print("strand_features after encoder", strand_features.mean(), strand_features.std())

        strand_features = strand_features.view(nr_strands, -1).contiguous()
        # strand_features = self.final_cnn_aggregator(strand_features) # outputs nr_strands x 128

        # print("strand_features after aggregate", strand_features.mean(), strand_features.std())

        strand_features_mean = self.aggregate_towards_mean(strand_features) # outputs nr_strands x 128
        s = self.pred_mean(strand_features_mean)
        
        # s = self.pred_mean(strand_features)
        # s=s

        # print("s mean and std", s.mean(), s.std())

        
        #pass the s through a tanh so it's bounded by -1,1 this makes it easier to consider it as an image later on when we train our diffusion model on scalp textures
        s=self.tanh(s)


        # exit(1)

        encoded_dict={}
        encoded_dict["z"]=s
        encoded_dict["z_no_eps"]=s

        if self.do_vae:
            s_mean = s
            # print("s_mean has mean std ", s_mean.mean(), s_mean.std())
            # s_logstd = 0.1 * self.pred_logstd(strand_features)
            strand_features_logstd = self.aggregate_towards_logstd(strand_features) # outputs nr_strands x 128
            s_logstd = -2.0 + 0.01*self.pred_logstd(strand_features_logstd) #start with logstd that is low so that initially the variance of the normal is also low
            encoded_dict["z_mean"] = s_mean
            encoded_dict["z_logstd"] = s_logstd
            # print("s_logstd has mean std ", s_logstd.mean(), s_logstd.std())
            if self.training:
                std = torch.exp(s_logstd)
                eps = torch.empty_like(std).normal_()
                # s = s + std * eps
                deviation = std * eps
                s = s + deviation
                encoded_dict["z"]=s
                # print("std std min max", std.min(), " ", std.max())
                # print("deviation mean std", deviation.mean(), " ", deviation.std())
                encoded_dict["z_deviation"] = deviation
                # print("strand std min max", std.min(), " ", std.max())
        
        return encoded_dict


class StrandGeneratorSiren(nn.Module):
   # a siren network which predicts various direction vectors along the strand similar
    def __init__(self, in_channels, modulation_hidden_dim, siren_hidden_dim, scale_init, decode_type, decode_random_verts, nr_verts_per_strand=256, nr_values_to_decode=256, dim_per_value_decoded=3):
        super(StrandGeneratorSiren, self).__init__()

        self.nr_verts_per_strand = nr_verts_per_strand
        self.nr_values_to_decode=nr_values_to_decode

        self.decode_type = decode_type
        self.decode_random_verts = decode_random_verts
    


        if self.decode_type=="xyz":
            nr_verts_to_create=self.nr_verts_per_strand
        elif self.decode_type=="dir":
            nr_verts_to_create = self.nr_verts_per_strand - 1 # we create only 99 because the frist one is just the origin
        else:
            raise ValueError("Unkown decode type: ", self.decode_type)

        if self.decode_random_verts:
            nr_verts_to_create = 1

        self.nr_verts_to_create=nr_verts_to_create

        self.activ = torch.nn.SiLU()
       

        self.nr_layers = 3
        cur_nr_channels = in_channels
        cur_nr_channels += 1
        # cur_nr_channels+=1 #+1 for the time t
        self.modulation_layers = torch.nn.ModuleList([])
        self.gain_per_layer = torch.nn.ParameterList([]) 
        # self.w_per_layer = torch.nn.ParameterList([]) 
        for i in range(self.nr_layers):
            self.modulation_layers.append(LinearWN_v2(cur_nr_channels, modulation_hidden_dim))
            cur_nr_channels = modulation_hidden_dim+in_channels +1  # at the end we concatenate the input z and a t
            self.gain_per_layer.append(torch.nn.Parameter(torch.ones([])))

        #not actually used during the forward pass but I have them so that the checkpoint still can load correctly
        self.second_modulation_layers = torch.nn.ModuleList([])
        for i in range(self.nr_layers):
            self.second_modulation_layers.append(LinearDummy(modulation_hidden_dim, modulation_hidden_dim))


        
        self.decode_val = LinearWN_v2(siren_hidden_dim, dim_per_value_decoded)
        self.gain_val = torch.nn.Parameter(torch.ones([siren_hidden_dim]))


        self.apply(lambda x: kaiming_init(x, False, nonlinearity="silu"))
        kaiming_init(self.decode_val, True)

        self.siren_layers = torch.nn.ModuleList([])
        self.siren_layers.append(BlockSiren(in_channels=1, out_channels=siren_hidden_dim, is_first_layer=True, scale_init=scale_init))
        for i in range(self.nr_layers-1):
            self.siren_layers.append(BlockSiren(in_channels=siren_hidden_dim, out_channels=siren_hidden_dim, scale_init=scale_init ))

        self.z_scaling = torch.nn.Parameter(torch.ones([])*0.2)
        self.hsiren_scaling = torch.nn.Parameter(torch.ones([]))


        #compute here a lot of the cuda stuff so we don't need to perform cpu-cuda transfers during the forward pass
        # sampling t
        self.t = torch.linspace(-1, 1, nr_verts_to_create).cuda() #between -1 and 1 because siren usually expects normalized input
        self.start_positions = torch.zeros(1, 1, 3).cuda()


    def forward(self, strand_features, hyperparams, normalization_dict):
        nr_strands = strand_features.shape[0]
        strand_features = strand_features.view(nr_strands, 1, -1).repeat(1, self.nr_verts_to_create, 1) # nr_strands x 100 x nr_channels
        t = self.t.view(1, self.nr_verts_to_create, -1).repeat(nr_strands, 1, 1) #nrstrands, nr_verts, nr_channels
        

        point_indices = None
        if self.decode_random_verts:
            # choose a random t for each strand
            # we can create only up until the very last vertex, except the tip, we need to be able to sample the next vertex so as to get a direction vector
            probability = torch.ones([nr_strands, self.num_pts - 2], dtype=torch.float32, device=torch.device("cuda")) 
            point_indices = torch.multinomial(probability, self.nr_verts_to_create, replacement=False) # size of the chunk size we selected
            # add also the next vertex on the strand so that we can compute directions
            point_indices = torch.cat([point_indices, point_indices + 1], 1)

            t = batched_index_select(t, 1, point_indices)

        # decode xyz
        h_siren = t
        z_scaling = self.z_scaling
        z = strand_features
        z_initial = z * z_scaling
        z = z * z_scaling

        #cat also T
        z=torch.cat([z,t],dim=2)


        hair_dir=None
        for i in range(self.nr_layers):
            gain = self.gain_per_layer[i]
            
            h_modulation = self.activ( self.modulation_layers[i](z))
           
            s = self.siren_layers[i](h_siren)
          
            #the input to the siren has to be unit gaussian, if we multiply by hmodulation, we are reducing the variance by Zscaling, so we boost the variance back up with this so that h_siren is unit gaussian again
            h_siren = h_modulation * s * (1.0/(z_scaling*gain)) 
            
            z = torch.cat([z_initial, h_modulation,t], 2)
            


        pred_dict={}

        if self.decode_type=="xyz":
            points_pos = self.decode_val(h_siren)
            if self.decode_random_verts:
                pred_strands = points_pos
            else:
                # start_positions = torch.zeros(nr_strands, 1, 3).cuda()
                start_positions = self.start_positions.repeat(nr_strands,1,1)
                pred_strands = torch.cat([start_positions, points_pos], 1)
                #positions are normalized to be in unit gaussian so we denormalize them to be in real space
                pred_strands=un_normalize_data(pred_strands, normalization_dict["xyz_mean"], normalization_dict["xyz_std"])
        elif self.decode_type=="dir":
            # divide by the nr of points on the strand otherwise the direction will have norm=1 and then when integrated you end up with a gigantic strand that has 100 units

            #predict dir
            hair_dir = self.decode_val(h_siren)
            
            #dirs are normalized to be in unit gaussian so we denormalize them to be in real space
            hair_dir=un_normalize_data(hair_dir, normalization_dict["dir_mean"], normalization_dict["dir_std"])
            pred_dict["strand_directions"]=hair_dir


            #predict pos 
            pred_strands = torch.cumsum(hair_dir, dim=1) # nr_strands, nr_verts-1, 3
            # we know that the first vertex is 0,0,0 so we just concatenate that one
            # start_positions = torch.zeros(nr_strands, 1, 3).cuda()
            start_positions = self.start_positions.repeat(nr_strands,1,1)
            pred_strands = torch.cat([start_positions, pred_strands], 1)


        # print("pred_strands", pred_strands.shape)
        pred_dict["strand_positions"]=pred_strands
        pred_dict["point_indices"]=point_indices
        return pred_dict
    

'''
uses only one Z tensor and predicts the strands using SIREN. There is no normalization apart from moving the strands to origin
is used to predict and regress only strand data, with no scalp
'''
class StrandCodec(nn.Module):
    def __init__(self, do_vae=True, scale_init=30.0, decode_type="dir", decode_random_verts=False, nr_verts_per_strand=256, nr_values_to_decode=255, dim_per_value_decoded=3):
        super(StrandCodec, self).__init__()

        self.do_vae = do_vae
        self.decode_type = decode_type
        self.decode_random_verts = decode_random_verts

        # encode
        
        self.encoder = StrandEncoder1dCNNWN(self.do_vae, out_channels=64) #Uses LinearWN
        
        # decoder
        self.decoder = StrandGeneratorSiren(in_channels=64, modulation_hidden_dim=128, siren_hidden_dim=128,
                                                     scale_init=scale_init, decode_type=decode_type, decode_random_verts=decode_random_verts,
                                                     nr_verts_per_strand=nr_verts_per_strand, nr_values_to_decode=nr_values_to_decode,
                                                     dim_per_value_decoded=dim_per_value_decoded,
                                                     )
        
       
                                                     
    def save(self, root_folder, experiment_name, hyperparams, iter_nr, info=None):
        name=str(iter_nr)
        if info is not None:
            name+="_"+info
        models_path = os.path.join(root_folder, experiment_name, name, "models")
        if not os.path.exists(models_path):
            os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "strand_codec.pt"))

        hyperparams_params_path=os.path.join(models_path, "hyperparams.json")
        with open(hyperparams_params_path, 'w', encoding='utf-8') as f:
            json.dump(vars(hyperparams), f, ensure_ascii=False, indent=4)


    def forward(self, gt_dict, hyperparams, normalization_dict):
        if hyperparams.normalize_input:
            gt_dict = normalize_gt_data(gt_dict, normalization_dict)
        encoded_dict = self.encoder(gt_dict)
        pred_dict=self.decoder(encoded_dict["z"], hyperparams, normalization_dict)
        return pred_dict, encoded_dict