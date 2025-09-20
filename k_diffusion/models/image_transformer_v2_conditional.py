"""k-diffusion transformer diffusion models, version 2."""

from dataclasses import dataclass
from functools import lru_cache, reduce
import math
from typing import Callable, Union

from einops import rearrange
from k_diffusion.models.modules import GlobalTransformerLayer, Level, Linear, LocalCondProj, NeighborhoodTransformerLayer, NoAttentionTransformerLayer, RMSNorm, ShiftedWindowTransformerLayer, TokenMerge, MappingNetwork, TokenSplit, TokenSplitWithoutSkip, downscale_pos, filter_params, tag_module
from .attention import SpatialTransformerSimpleV2
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F
import sys
import os


from . import flags, flops
from .. import layers
from .axial_rope import make_axial_pos

# from difflocks_modules.networks import LinearWN_v2, Conv1dWN_v2, BlockSiren, kaiming_init
from difflocks_modules.edm2_modules import MPFourier
# from difflocks_modules.edm2_modules import  mp_silu



# Configuration

@dataclass
class GlobalAttentionSpec:
    d_head: int


@dataclass
class NeighborhoodAttentionSpec:
    d_head: int
    kernel_size: int


@dataclass
class ShiftedWindowAttentionSpec:
    d_head: int
    window_size: int


@dataclass
class NoAttentionSpec:
    pass


@dataclass
class LevelSpec:
    depth: int
    width: int
    d_ff: int
    self_attn: Union[GlobalAttentionSpec, NeighborhoodAttentionSpec, ShiftedWindowAttentionSpec, NoAttentionSpec]
    dropout: float


@dataclass
class MappingSpec:
    depth: int
    width: int
    d_ff: int
    dropout: float


# Model class

class ImageTransformerDenoiserModelV2Conditional(nn.Module):
    def __init__(self, levels, mapping, in_channels, out_channels, patch_size, input_size, condition_dropout_rate, rgb_condition_config, num_classes=0, mapping_cond_dim=0, do_multires=False):
        super().__init__()
        self.num_classes = num_classes
        self.do_multires=do_multires
        self.condition_dropout_rate=condition_dropout_rate
        self.rgb_condition_config=rgb_condition_config

        self.patch_in = TokenMerge(in_channels+16, levels[0].width, patch_size)
        # self.patch_in = TokenMergeEDM2(in_channels, levels[0].width, patch_size)

        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.time_in_proj_only_t = Linear(mapping.width, mapping.width, bias=False)
        self.mapping = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")
        self.mapping_only_t = tag_module(MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping")
       

        #rgb_featrues
        self.cross_down_layer = nn.ModuleList()
        self.cross_mid_layer = nn.ModuleList()
        self.cross_up_layer = nn.ModuleList()

      

        #dino global cond
        print("global_condition_shape dim1", rgb_condition_config['global_condition_shape'][1]*2)
        self.global_latent_encoder = nn.Sequential(
            nn.Linear(rgb_condition_config['global_condition_shape'][1]*2, mapping.width, bias=True),
        )

        
        self.up_local_proj_list=nn.ModuleList()
        self.mid_local_proj_list=None
        self.down_local_proj_list=nn.ModuleList()
       
        self.local_proj= LocalCondProj(rgb_condition_config["local_condition_shapes"][0]["shape"][1], rgb_condition_config["cross_condition_dim"][0], mapping.width)
       
        self.down_levels, self.up_levels = nn.ModuleList(), nn.ModuleList()
       

        for lvl in levels:
            print("width for this lvl", lvl.width)

        for i, spec in enumerate(levels):
            in_channels = spec.width
            print("initializing LVL ", i, " with in_channels", in_channels)
            if isinstance(spec.self_attn, GlobalAttentionSpec):
                layer_factory = lambda _: GlobalTransformerLayer(in_channels, spec.d_ff, spec.self_attn.d_head, mapping.width, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NeighborhoodAttentionSpec):
                layer_factory = lambda _: NeighborhoodTransformerLayer(in_channels, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.kernel_size, dropout=spec.dropout)
            elif isinstance(spec.self_attn, ShiftedWindowAttentionSpec):
                layer_factory = lambda i: ShiftedWindowTransformerLayer(in_channels, spec.d_ff, spec.self_attn.d_head, mapping.width, spec.self_attn.window_size, i, dropout=spec.dropout)
            elif isinstance(spec.self_attn, NoAttentionSpec):
                layer_factory = lambda _: NoAttentionTransformerLayer(in_channels, spec.d_ff, mapping.width, dropout=spec.dropout)
            else:
                raise ValueError(f"unsupported self attention spec {spec.self_attn}")


            if i < len(levels) - 1:
                print("spec_depth is", spec.depth)
                self.down_levels.append(Level([layer_factory(i) for i in range(spec.depth)]))
                self.up_levels.append(Level([layer_factory(i + spec.depth) for i in range(spec.depth)]))
                print("nr downlvl",len(self.down_levels))

                #for rgb cross attention
                d_head = 64 #TODO make this a parameter in the cofnig file
                n_heads = in_channels//d_head
                print("making up cross layer nr ", i)
                up_condition_dim=rgb_condition_config["cross_condition_dim"][i]
                up_do_self_attn=rgb_condition_config["self_attn"][i]
                print("up_do_self_attn", up_do_self_attn)

                down_condition_dim=rgb_condition_config["cross_condition_dim"][i]
                down_do_self_attn=rgb_condition_config["self_attn"][i]
                print("down_do_self_attn", down_do_self_attn)
                print("down_condition_dim", down_condition_dim)
               

                self.cross_up_layer.append(SpatialTransformerSimpleV2(in_channels, n_heads, d_head, 
                                                                       global_cond_dim=mapping.width,
                                                             context_dim=up_condition_dim,
                                                             do_self_attention=up_do_self_attn,
                                                             dropout=0.0))
                self.cross_down_layer.append(SpatialTransformerSimpleV2(in_channels, n_heads, d_head, 
                                                                       global_cond_dim=mapping.width,
                                                             context_dim=down_condition_dim,
                                                             do_self_attention=down_do_self_attn,
                                                             dropout=0.0))




            else:
                self.mid_level = Level([layer_factory(i) for i in range(spec.depth)])

                #for rgb cross attention
                d_head = 64 #TODO make this a parameter in the cofnig file
                n_heads = in_channels//d_head
                mid_condition_dim=rgb_condition_config["cross_condition_dim"][-1]
                mid_do_self_attn=rgb_condition_config["self_attn"][-1]
                print(" init mid_condition_dim", mid_condition_dim)

                self.cross_mid_layer.append(SpatialTransformerSimpleV2(in_channels, n_heads, d_head, 
                                                                       global_cond_dim=mapping.width,
                                                             context_dim=mid_condition_dim,
                                                             do_self_attention=mid_do_self_attn,
                                                             dropout=0.0))


      


        self.merges = nn.ModuleList([TokenMerge(spec_1.width, spec_2.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
        self.splits = nn.ModuleList([TokenSplit(spec_2.width, spec_1.width) for spec_1, spec_2 in zip(levels[:-1], levels[1:])])
      

        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)

        #from edm2
        logvar_channels = 128
        self.logvar_fourier = MPFourier(logvar_channels)
        # self.logvar_linear = MPLinear(logvar_channels, 1, bias=False)
        self.logvar_linear = Linear(logvar_channels, 1, bias=False)
        nn.init.zeros_(self.logvar_linear.weight)

       
        self.untied_bias= nn.Parameter(torch.zeros((1,16,input_size[0], input_size[1])))
        
        
       

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, sigma, latents_dict=None, aug_cond=None, class_cond=None, mapping_cond=None, cross_cond=None, cam=None):
        #get noise only embedding
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj_only_t(self.time_emb(c_noise[..., None]))
        noise_cond = self.mapping_only_t(time_emb)


       
        ################################# starting with latents
        #sometimes drop the conditioning in order do CFG
        with torch.no_grad():
            nr_batches = x.shape[0]
            rand=torch.rand((nr_batches), device=x.device) #rand between 0,1
            cond_batches_to_drop=(rand<self.condition_dropout_rate)*1.0 #0.1 batches are set to true
            cond_batches_to_keep = 1.0-cond_batches_to_drop #0.9 here are ones and 0.1 are set to zero
            
            #gather here the local and global conditioning for each lvl, or make dummy values if we didn't provide anything and therefore we are doing unconditional generation
            global_cond=None
            locals_cond_list=[]
            if latents_dict is None:
                global_cond=torch.zeros((1,self.rgb_condition_config["global_condition_shape"][1]*2), device=x.device)
                for local_shape in self.rgb_condition_config["local_condition_shapes"]:
                    locals_cond_list.append(torch.zeros(local_shape["shape"], device=x.device))
            else:
                #populate with latents_dict
                #global
                dino_latent_mean=latents_dict["dinov2"]["final_latent"].mean(dim=[2,3])
                dino_latent_cls=latents_dict["dinov2"]["cls_token"]
                global_cond=torch.cat([dino_latent_mean,dino_latent_cls],1)
                #locals
                local=latents_dict["dinov2"]["final_latent"].contiguous() #needed otherwise ddp complains about gradients being not contiguous and that affects performance
                for local_shape in self.rgb_condition_config["local_condition_shapes"]:
                    locals_cond_list.append(local)


            #condition dropout
            global_cond=global_cond*cond_batches_to_keep.view(nr_batches,1)
            for l_idx in range(len(locals_cond_list)):
                locals_cond_list[l_idx]=locals_cond_list[l_idx]*cond_batches_to_keep.view(nr_batches,1,1,1)

            #make position embedding for the locals
            locals_pos_list=[]
            for local_cond in locals_cond_list:
                pos_img = make_axial_pos(local_cond.shape[-1], local_cond.shape[-2], device=x.device).view(local_cond.shape[-1], local_cond.shape[-2], 2)
                locals_pos_list.append(pos_img)  
        #########################finished with the latents
                
       

        #concat one to act as a bias
        x = torch.cat([x, self.untied_bias.repeat(x.shape[0],1,1,1) ], dim=1)
       

        # Patching
        x = x.movedim(-3, -1)
        x = self.patch_in(x)
        # TODO: pixel aspect ratio for nonsquare patches
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device).view(x.shape[-3], x.shape[-2], 2) #[64, 64, 2]
      


        # Mapping network
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        global_cond_embedded=self.global_latent_encoder(global_cond)

        embedding_summed=time_emb + global_cond_embedded
        cond = self.mapping(embedding_summed)

        #maps locals only once
        local_cond = locals_cond_list[0]
        local_cond = self.local_proj(local_cond, noise_cond)

        # Hourglass transformer
        skips, poses = [], []
        for i, (down_level, merge) in enumerate(zip(self.down_levels, self.merges)):
            x = down_level(x, pos, cond)
           

            ####CROSS CONDITION
            x = x.movedim(-1, -3)
            local_pos = locals_pos_list[i]
            x = self.cross_down_layer[i](x, pos, cond, local_cond, local_pos)
            x = x.movedim(-3, -1)

            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        x = self.mid_level(x, pos, cond)
        

        ####CROSS CONDITION
        x = x.movedim(-1, -3)
        local_pos = locals_pos_list[-1]
        x = self.cross_mid_layer[0](x, pos, cond, local_cond, local_pos)
        x = x.movedim(-3, -1)

       

        nr_up_lvls=len(self.up_levels)
        lvl_idx=0
        for i,(up_level, split, skip, pos) in enumerate(reversed(list(zip(self.up_levels, self.splits, skips, poses)))):
            x = split(x, skip)
            x = up_level(x, pos, cond)
            

            ####CROSS CONDITION
            x = x.movedim(-1, -3)
            local_pos = locals_pos_list[-i-1]
            x = self.cross_up_layer[-i-1](x, pos, cond, local_cond, local_pos)
            x = x.movedim(-3, -1)
            lvl_idx+=1



        # Unpatching
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)

        logvar = 1.0+self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1,1,1,1)

        return x, logvar
