from modules.difflocks.difflocks_modules.networks import kaiming_init
import torch
from torch import nn
import os
import json
import numpy as np


class RGB2MaterialModel(nn.Module):

    def __init__(self, input_dim, out_dim, hidden_dim):
        super().__init__()

        self.out_dim=out_dim



        #attempt 2
        self.dino2conf=nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, padding=0, bias=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.dino2mat=nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, padding=0, bias=True),
            nn.SiLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
            

        

        self.apply(lambda x: kaiming_init(x, False, nonlinearity="silu"))

    def save(self, root_folder, experiment_name, hyperparams, iter_nr, info=None):
        name=str(iter_nr)
        if info is not None:
            name+="_"+info
        models_path = os.path.join(root_folder, experiment_name, name, "models")
        if not os.path.exists(models_path):
            os.makedirs(models_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(models_path, "rgb2material.pt"))

        hyperparams_params_path=os.path.join(models_path, "hyperparams.json")
        with open(hyperparams_params_path, 'w', encoding='utf-8') as f:
            json.dump(vars(hyperparams), f, ensure_ascii=False, indent=4)


    def forward(self, batch_dict):

        x=batch_dict["dinov2_latents"] #BCHW (1,1024,55,55)


        #attempt 2, each patch predicts a confidence and a material, then we average all the materials across all patches, weighted by confidence
        conf=self.dino2conf(x)
        mat=self.dino2mat(x)
        

        #average the mat across the pixels
        avg_mat = (mat*conf).sum((2,3)) / (conf.sum((2,3)) +1e-6) #sum across all H and W dimensions
        x=avg_mat


        #split the material in parameters, at least the ones that are meaningfull and actually have a loss applied to them
        melanin=x[:,3]
        redness=x[:,4]
        root_darkness_start=x[:,8]
        root_darkness_end=x[:,9]
        root_darkness_strength=x[:,10]



        pred_dict={}
        pred_dict["material"]=x
        pred_dict["melanin"]=melanin
        pred_dict["redness"]=redness
        pred_dict["root_darkness_start"]=root_darkness_start
        pred_dict["root_darkness_end"]=root_darkness_end
        pred_dict["root_darkness_strength"]=root_darkness_strength
       

        return pred_dict