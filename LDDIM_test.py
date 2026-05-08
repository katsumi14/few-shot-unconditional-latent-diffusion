import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
import torch
import numpy as np
from tqdm import tqdm
from DDPM_networks import NoiseScheduler,LDDPM_VAE, LDDPM_UNet
from mylib.data_io import show_images, to_sigmoid_image
from mylib.utility import print_args


# generate images by LDDIM sampling
def LDDIM_generate(Z, model, vae_model, noise_scheduler, n_timesteps, n_gen_timesteps, save_progress=False):

    t_list = np.round(np.linspace(0, n_timesteps-1, n_gen_timesteps)).astype(np.int32)
    s_list = np.concatenate([[0], t_list[:-1]])
    timesteps = np.concatenate([t_list.reshape(-1, 1), s_list.reshape(-1, 1)], axis=1)
    with torch.no_grad():
        for t_idx, s_idx in tqdm(reversed(timesteps), total=n_gen_timesteps):

            # predict noise
            t = t_idx * torch.ones((len(Z),), device=Z.device).long()
            noise = model(Z, t)

            # remove noise
            if t_idx == 0:
                Z = noise_scheduler.sqrt_inv_alpha[t_idx] * Z - noise_scheduler.noise_scale_coeff[t_idx] * noise
            else:
                Z = (noise_scheduler.sqrt_alpha_bar[s_idx] / noise_scheduler.sqrt_alpha_bar[t_idx]) * (Z - noise_scheduler.sqrt_one_minus_alpha_bar[t_idx] * noise)
                Z = Z + noise_scheduler.sqrt_one_minus_alpha_bar[s_idx] * noise

            # save the progress of denoising
            if save_progress:
                Y = vae_model.decode(Z)
                Y_cpu = to_sigmoid_image(Y).to('cpu').detach()
                show_images(Y_cpu, num=len(Y), num_per_row=8, title='timestep_{}'.format(t_idx+1), save_only=True, save_dir=MODEL_DIR)
    
    return Z

