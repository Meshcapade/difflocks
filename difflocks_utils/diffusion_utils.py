import torch
import k_diffusion as K

@torch.no_grad()
def sample_images(nr_images, model_ema, model_config, nr_iters=100, extra_args={}, callback=None):
    model_ema.eval()
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    size = model_config['input_size']
    n_per_proc = nr_images
    x = torch.randn([1, n_per_proc, model_config['input_channels'], size[0], size[1]]).cuda()
    x = x[0] * sigma_max
    model_fn = model_ema
    sigmas = K.sampling.get_sigmas_karras(nr_iters, sigma_min, sigma_max, rho=7., device="cuda")
    x_0 = K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=False, callback=callback)
    return x_0

@torch.no_grad()
#samples using classifier free guidance and only enables the cfg_val when the sigma is within the interval.
#idead from this paper: https://arxiv.org/pdf/2404.07724
def sample_images_cfg(nr_images, cfg_val, cfg_interval, model_ema, model_config, nr_iters=100, extra_args={}, callback=None):
    model_ema.eval()
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    size = model_config['input_size']
    n_per_proc = nr_images
    x = torch.randn([1, n_per_proc, model_config['input_channels'], size[0], size[1]]).cuda()
    x = x[0] * sigma_max
    model_fn = model_ema
    sigmas = K.sampling.get_sigmas_karras(nr_iters, sigma_min, sigma_max, rho=7., device="cuda")
    x_0 = K.sampling.sample_dpmpp_2m_sde_cfg(model_fn, x, sigmas, cfg_val, cfg_interval, extra_args=extra_args,  eta=0.0, solver_type='heun', disable=False, callback=callback)
    return x_0