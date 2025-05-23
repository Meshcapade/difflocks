from functools import lru_cache, reduce
import math

from dctorch import functional as df
from einops import rearrange, repeat
import torch
from torch import nn
from torch.nn import functional as F

from . import sampling, utils


# Helper functions


def dct(x):
    if x.ndim == 3:
        return df.dct(x)
    if x.ndim == 4:
        return df.dct2(x)
    if x.ndim == 5:
        return df.dct3(x)
    raise ValueError(f'Unsupported dimensionality {x.ndim}')


@lru_cache
def freq_weight_1d(n, scales=0, dtype=None, device=None):
    ramp = torch.linspace(0.5 / n, 0.5, n, dtype=dtype, device=device)
    weights = -torch.log2(ramp)
    if scales >= 1:
        weights = torch.clamp_max(weights, scales)
    return weights


@lru_cache
def freq_weight_nd(shape, scales=0, dtype=None, device=None):
    indexers = [[slice(None) if i == j else None for j in range(len(shape))] for i in range(len(shape))]
    weights = [freq_weight_1d(n, scales, dtype, device)[ix] for n, ix in zip(shape, indexers)]
    return reduce(torch.minimum, weights)


# Karras et al. preconditioned denoiser


class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, sigma_data=1., weighting='karras', scales=1, parametrization="v", loss_weight_per_channel=None):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.scales = scales
        self.parametrization = parametrization
        if callable(weighting):
            self.weighting = weighting
        if weighting == 'karras':
            self.weighting = torch.ones_like
        elif weighting == 'soft-min-snr':
            self.weighting = self._weighting_soft_min_snr
        elif weighting == 'snr':
            self.weighting = self._weighting_snr
        # elif weighting == 'edm2':
            # self.weighting = self._weighting_edm2
        else:
            raise ValueError(f'Unknown weighting type {weighting}')

        if loss_weight_per_channel is not None: 
            self.loss_weight_per_channel=torch.as_tensor(loss_weight_per_channel).cuda()
        else:
            self.loss_weight_per_channel=None
        
        

    def _weighting_soft_min_snr(self, sigma):
        return (sigma * self.sigma_data) ** 2 / (sigma ** 2 + self.sigma_data ** 2) ** 2

    def _weighting_snr(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def _weighting_edm2(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        # print("Denoiser")
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        # print("sigma",sigma)
        c_weight = self.weighting(sigma)
        # print("c_weight is ", c_weight)
        # edm2_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        weight = c_weight
        # weight = torch.ones_like(sigma)


        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        # model_output, multires_output, logvar = self.inner_model(noised_input * c_in, sigma, **kwargs)
        if 'step' in kwargs:
            step = kwargs['step']
            del kwargs['step']
        result = self.inner_model(noised_input * c_in, sigma, **kwargs)
        if len(result)==2:
            # model_output, multires_output, logvar = result
            model_output, logvar = result
        elif len(result)==4:

            model_output, multires_output, logvar,clip_feature_embedding = result

        #if you have channel weights they are usually computed from the script create_strand_latent_weights.py
        if self.loss_weight_per_channel is None:
            loss_weight_per_channel=torch.ones(model_output.shape[1], device=model_output.device)
        else:
            loss_weight_per_channel=self.loss_weight_per_channel
        loss_weight_per_channel=loss_weight_per_channel.view(1,-1,1,1)


        #loss for regions with zero density can be set to zero
        density_map=input[:,-1:,:,:]
        density_map=density_map*(0.5/self.sigma_data) + 0.5 #after training on lambda
        density_map=density_map.clamp(0, 1)
        #low density regions
        low_density=density_map<0.001
        loss_weight_spatial=torch.ones_like(density_map, device=input.device)
        loss_weight_spatial[low_density]=0.0
        loss_weight_spatial=loss_weight_spatial.repeat(1,model_output.shape[1],1,1)
        loss_weight_spatial[:,-1:,:,:]=1.0 #density weight is all ones
        # print("loss_weight_spatial",loss_weight_spatial.shape)
        # print("model_output",model_output.shape)

   


  

        if self.parametrization =="v":
            #the original loss used in k-diffusion
            target = (input - c_skip * noised_input) / c_out
            mse_loss=(loss_weight_spatial*loss_weight_per_channel*(( (model_output.to(torch.float32)*c_out + noised_input*c_skip) - input) ** 2)).flatten(1).mean(1)
        elif self.parametrization =="x0":
            #directly predict target
            target=input
            mse_loss=(loss_weight_spatial*loss_weight_per_channel*(( model_output.to(torch.float32)  - input) ** 2)).flatten(1).mean(1)
        # print("target mean and std", target.mean(), target.std())
        # print("output mean and std", model_output.mean(), model_output.std())

        # print("model_output",model_output.shape)


        #more similar to edm2 
        # model_output= c_skip * noised_input + c_out * model_output.to(torch.float32)
        # target=input
        # mse_loss=((model_output - target) ** 2).flatten(1).mean(1)

        #directly predict target
        # target=input
        # mse_loss=(( model_output.to(torch.float32)  - input) ** 2).flatten(1).mean(1)


        loss = None
        singleres_loss=None
        multires_loss=torch.zeros([input.shape[0]],device=input.device)
        if self.scales == 1:
            # loss= ((model_output - target) ** 2).flatten(1).mean(1)  * weight

           

            #weightign each channel differently           
            loss = (weight.view(-1,1,1,1) * loss_weight_per_channel.view(1,-1,1,1) / logvar.exp()) * ((model_output - target) ** 2) + logvar

            # loss = (weight.view(-1,1,1,1) / logvar.exp()) * ((model_output - target) ** 2) + logvar
            # print("effective weight", (weight.view(-1,1,1,1) / logvar.exp()).flatten()  )
            loss=loss.flatten(1).mean(1)
            singleres_loss=loss.clone()


            # if multires_output is not None:
            #     #do also a multires loss
            #     for idx_mr, mr_out in enumerate(reversed(multires_output)):
            #         # print("idx_mr", idx_mr)
            #         # print("mr_out", mr_out.shape)
            #         mr_weight = 1.0/ (np.power(2,idx_mr+1) * np.power(2,idx_mr+1))
            #         # print("mr_weight",mr_weight)
            #         with torch.no_grad():
            #             target_mr = resize_right.resize(target, out_shape=[mr_out.shape[-1], mr_out.shape[-2]], interp_method=interp_methods.linear)
            #         # print("target_mr",target_mr.shape)
            #         cur_loss_mr= ((mr_out - target_mr) ** 2).flatten(1).mean(1) * c_weight * mr_weight
            #         loss+=cur_loss_mr
            #         multires_loss+=cur_loss_mr

            if len(result)==4:
                if clip_feature_embedding is not None:
                    gt_clip_feature = kwargs['latent_dict']['clip']['ClipImageFeature']

                    clip_similar_loss =(1-torch.cosine_similarity(clip_feature_embedding,gt_clip_feature[:,0],dim=-1)).mean()
                    if step<50000:
                        weight = 0.1
                    else:
                        weight = max(0.1* 0.3**(step//50000),0.003) ### also can set 1, set 50 for experiments
                    # weight = 0
                    # weight = 0
                    loss += clip_similar_loss * weight
                    loss+= F.l1_loss(clip_feature_embedding,gt_clip_feature[:,0])*weight
                else:
                    clip_similar_loss = None
                return loss, singleres_loss, multires_loss, mse_loss,clip_similar_loss

            return loss, singleres_loss, multires_loss, mse_loss
        sq_error = dct(model_output - target) ** 2
        f_weight = freq_weight_nd(sq_error.shape[2:], self.scales, dtype=sq_error.dtype, device=sq_error.device)
        return (sq_error * f_weight).flatten(1).mean(1) * c_weight

    def forward(self, input, sigma, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        # denoised, _, _ = self.inner_model(input * c_in, sigma, **kwargs)
        denoised = self.inner_model(input * c_in, sigma, **kwargs)[0]
        # return denoised.to(torch.float32) * c_out + input * c_skip
        # return denoised.to(torch.float32)
        if self.parametrization =="v":
            return denoised.to(torch.float32) * c_out + input * c_skip
        elif self.parametrization =="x0":
            #directly predicts the clean image
            return denoised.to(torch.float32)


class DenoiserWithVariance(Denoiser):
    def loss(self, input, noise, sigma, **kwargs):
        print("DenoiserWithVariance")
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output, logvar = self.inner_model(noised_input * c_in, sigma, return_variance=True, **kwargs)
        logvar = utils.append_dims(logvar, model_output.ndim)
        target = (input - c_skip * noised_input) / c_out
        losses = ((model_output - target) ** 2 / logvar.exp() + logvar) / 2
        return losses.flatten(1).mean(1)


class SimpleLossDenoiser(Denoiser):
    """L_simple with the Karras et al. preconditioner."""

    def loss(self, input, noise, sigma, **kwargs):
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        denoised = self(noised_input, sigma, **kwargs)
        eps = sampling.to_d(noised_input, sigma, denoised)
        return (eps - noise).pow(2).flatten(1).mean(1)


# Residual blocks

class ResidualBlock(nn.Module):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


# Noise level (and other) conditioning

class ConditionedModule(nn.Module):
    pass


class UnconditionedModule(ConditionedModule):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, cond=None):
        return self.module(input)


class ConditionedSequential(nn.Sequential, ConditionedModule):
    def forward(self, input, cond):
        for module in self:
            if isinstance(module, ConditionedModule):
                input = module(input, cond)
            else:
                input = module(input)
        return input


class ConditionedResidualBlock(ConditionedModule):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = ConditionedSequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input, cond):
        skip = self.skip(input, cond) if isinstance(self.skip, ConditionedModule) else self.skip(input)
        return self.main(input, cond) + skip


class AdaGN(ConditionedModule):
    def __init__(self, feats_in, c_out, num_groups, eps=1e-5, cond_key='cond'):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.cond_key = cond_key
        self.mapper = nn.Linear(feats_in, c_out * 2)
        nn.init.zeros_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)

    def forward(self, input, cond):
        weight, bias = self.mapper(cond[self.cond_key]).chunk(2, dim=-1)
        input = F.group_norm(input, self.num_groups, eps=self.eps)
        return torch.addcmul(utils.append_dims(bias, input.ndim), input, utils.append_dims(weight, input.ndim) + 1)


# Attention


class SelfAttention2d(ConditionedModule):
    def __init__(self, c_in, n_head, norm, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        self.norm_in = norm(c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input, cond):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm_in(input, cond))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class CrossAttention2d(ConditionedModule):
    def __init__(self, c_dec, c_enc, n_head, norm_dec, dropout_rate=0.,
                 cond_key='cross', cond_key_padding='cross_padding'):
        super().__init__()
        assert c_dec % n_head == 0
        self.cond_key = cond_key
        self.cond_key_padding = cond_key_padding
        self.norm_enc = nn.LayerNorm(c_enc)
        self.norm_dec = norm_dec(c_dec)
        self.n_head = n_head
        self.q_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.kv_proj = nn.Linear(c_enc, c_dec * 2)
        self.out_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input, cond):
        n, c, h, w = input.shape
        q = self.q_proj(self.norm_dec(input, cond))
        q = q.view([n, self.n_head, c // self.n_head, h * w]).transpose(2, 3)
        kv = self.kv_proj(self.norm_enc(cond[self.cond_key]))
        kv = kv.view([n, -1, self.n_head * 2, c // self.n_head]).transpose(1, 2)
        k, v = kv.chunk(2, dim=1)
        attn_mask = (cond[self.cond_key_padding][:, None, None, :]) * -10000
        y = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


# Downsampling/upsampling

_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic': 
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
        0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3': 
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
        -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
        0.44638532400131226, 0.13550527393817902, -0.066637322306633,
        -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}
_kernels['bilinear'] = _kernels['linear']
_kernels['bicubic'] = _kernels['cubic']


class Downsample2d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv2d(x, weight, stride=2)


class Upsample2d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv_transpose2d(x, weight, stride=2, padding=self.pad * 2 + 1)


# Embeddings

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


# U-Nets

class UNet(ConditionedModule):
    def __init__(self, d_blocks, u_blocks, skip_stages=0):
        super().__init__()
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(u_blocks)
        self.skip_stages = skip_stages

    def forward(self, input, cond):
        skips = []
        for block in self.d_blocks[self.skip_stages:]:
            input = block(input, cond)
            skips.append(input)
        for i, (block, skip) in enumerate(zip(self.u_blocks, reversed(skips))):
            input = block(input, cond, skip if i > 0 else None)
        return input
