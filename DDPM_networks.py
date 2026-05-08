import torch
import torch.nn as nn
import torch.nn.functional as F
from DDPM_modules import *

#GRL layer
class GradientReversalFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_forward: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)
        return input_forward
 
    @staticmethod
    def backward(ctx, grad_backward: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, = ctx.saved_tensors
        return scale * -grad_backward, None

class GradientReversal(nn.Module):
 
    def __init__(self, scale: float):
        super(GradientReversal, self).__init__()
        self.scale = torch.tensor(scale)
 
    def forward(self, x: torch.Tensor,scale) -> torch.Tensor:
        self.scale=torch.tensor(scale)
        return GradientReversalFunction.apply(x, self.scale)

# NoiseScheduler
class NoiseScheduler:

    def __init__(self, device, method:str='linear', num_timesteps:int=1000, start:float=0.0001, end:float=0.02, s:float=0.008, clip:float=0.999):

        # prepare the beta schedule
        if method == 'cosine': 
            num_timesteps += 1
            T = num_timesteps - 1
            t = torch.arange(0, num_timesteps)
            alpha_bar = torch.cos(0.5 * torch.pi * ((t/T)+s)/(1+s))**2
            alpha_bar = alpha_bar / alpha_bar[0]
            beta = torch.clamp(1.0 - alpha_bar[1:] / alpha_bar[:-1], max=clip)
        elif method == 'quadratic': 
            beta = torch.linspace(start**0.5, end**0.5, num_timesteps)**2
        elif method == 'sigmoid': 
            beta = torch.sigmoid(torch.linspace(-6, 6, num_timesteps)) * (end - start) + start
        elif method == 'linear': 
            beta = torch.linspace(start, end, num_timesteps)
        else:
            raise NotImplementedError(method)
        self.beta = beta.to(device)

        # prepare the alpha, alpha_bar, etc.
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, axis=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_inv_alpha = torch.sqrt(1.0 / self.alpha)

        # prepare the coefficients used in the reverse diffusion process
        self.var_coeff = torch.sqrt(self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar))
        self.noise_scale_coeff = self.sqrt_inv_alpha * self.beta / self.sqrt_one_minus_alpha_bar

    # At the step of time t, generate a noisy sample by adding Gaussian noise to x0
    # -x0: the input image before adding noise (given in mini-batch format)
    # -t: the time step (given in mini-batch format)
    # -noise: the seed noise following the standard normal distribution (if None, it will be generated within the function)
    def get_noisy_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1) * x0 + self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1) * noise


# The VAE for latent diffusion model
# The VAE encoder embeds (C,H,W) images into (ZC,H/4,W/4) latent feature maps
# The VAE decoder reconstructs (C,H,W) images from (ZC,H/4,W/4) latent feature maps
class LDDPM_VAE(nn.Module):

    # C: the number of channels in the input image (1 for grayscale images, 3 for color images)
    # ZC: the number of channels in the latent feature map
    # num_groups: the number of groups in Group Normalization
    def __init__(self, C, ZC,label_embed_dim,num_groups=4):
        super(LDDPM_VAE, self).__init__()

        # the number of channels in each layer
        L1_C = 8
        L2_C = L1_C*2
        L3_C = L1_C*4
        L4_C = L1_C*8
        L5_C = L1_C*8

        # Enocoder
        self.init_conv = nn.Conv2d(in_channels=C, out_channels=L1_C, kernel_size=1, stride=1, padding=0)
        self.downsample = nn.Sequential(
            DDPMDownSamplingLayer(in_channels=L1_C, out_channels=L2_C, time_embed_dim=0, num_groups=num_groups, with_skip_output=False),
            DDPMDownSamplingLayer(in_channels=L2_C, out_channels=L3_C, time_embed_dim=0, num_groups=num_groups, with_skip_output=False),
            DDPMDownSamplingLayer(in_channels=L3_C, out_channels=L4_C, time_embed_dim=0, num_groups=num_groups, with_skip_output=False),
            DDPMDownSamplingLayer(in_channels=L4_C, out_channels=L5_C, time_embed_dim=0, num_groups=num_groups, with_skip_output=False, with_downsample=False),
            PreNormConv2d(in_channels=L4_C, out_channels=ZC*2, num_groups=num_groups, kernel_size=1, stride=1, padding=0),
        )
        self.to_mu = nn.Conv2d(in_channels=ZC*2, out_channels=ZC, kernel_size=1, stride=1, padding=0)
        self.to_lnvar = nn.Conv2d(in_channels=ZC*2, out_channels=ZC, kernel_size=1, stride=1, padding=0)
        #Label Classifier
        self.dis=nn.Sequential(
             nn.Conv2d(in_channels=ZC,out_channels=ZC*4,kernel_size=4,stride=2,padding=1),
             nn.BatchNorm2d(16),
             nn.ReLU(),
             
             nn.Conv2d(in_channels=ZC*4,out_channels=ZC*8,kernel_size=4,stride=2,padding=1),
             nn.BatchNorm2d(32),
             nn.ReLU(),
             
             nn.AdaptiveAvgPool2d(1),
             nn.Flatten(),
             nn.Linear(32,2)
             )
        self.grl=GradientReversal(scale=1.0)
        # Decoder
        self.label_embed= nn.Embedding(2,label_embed_dim)
        self.upsample0 = nn.Conv2d(in_channels=ZC, out_channels=L4_C, kernel_size=1, stride=1, padding=0)
        self.upsample1 = DDPMUpSamplingLayer_Alt(in_channels=L5_C, out_channels=L4_C, label_embed_dim=label_embed_dim, num_groups=num_groups, with_skip_input=False, with_upsample=False)
        self.upsample2 = DDPMUpSamplingLayer_Alt(in_channels=L4_C, out_channels=L3_C, label_embed_dim=label_embed_dim, num_groups=num_groups, with_skip_input=False)
        self.upsample3 = DDPMUpSamplingLayer_Alt(in_channels=L3_C, out_channels=L2_C, label_embed_dim=label_embed_dim, num_groups=num_groups, with_skip_input=False)
        self.upsample4 = DDPMUpSamplingLayer_Alt(in_channels=L2_C, out_channels=L1_C, label_embed_dim=label_embed_dim, num_groups=num_groups, with_skip_input=False)
        self.last_conv = nn.Sequential(
            PreNormConv2d(in_channels=L1_C, out_channels=L1_C, num_groups=num_groups, kernel_size=3, stride=1, padding=1),
            PreNormConv2d(in_channels=L1_C, out_channels=C, num_groups=num_groups, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
    # Encode
    def encode(self, x, testmode=False):
        h = self.init_conv(x)
        h = self.downsample(h)
        mu = self.to_mu(h)
        if testmode:
            return mu # when the model is in test mode, the mu (without noise) is returned as the output of the encoder
        else:
            lnvar = self.to_lnvar(h)
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * lnvar)
            return z, mu, lnvar
    #　Classify
    def discriminate(self,z,grl_scale):
        h=self.grl(z,grl_scale)
        y=self.dis(h)
        return y
    
    # Decode
    def decode(self, z,label):
        label_embedding= self.label_embed(label)
        h = self.upsample0(z)
        h = self.upsample1(h,label_embedding=label_embedding)
        h = self.upsample2(h,label_embedding=label_embedding)
        h = self.upsample3(h,label_embedding=label_embedding)
        h = self.upsample4(h,label_embedding=label_embedding)
        y = self.last_conv(h)
        return y

    # Decode (save outputs from intermediate layers)
    def decode_with_hidden_outputs(self, z,label):
        label_embedding=self.label_embed(label)
        h0 = self.upsample0(z)
        h1 = self.upsample1(h0,label_embedding=label_embedding)
        h2 = self.upsample2(h1,label_embedding=label_embedding)
        h3 = self.upsample3(h2,label_embedding=label_embedding)
        h4 = self.upsample4(h3,label_embedding=label_embedding)
        y = self.last_conv(h4)
        return y, h4, h3, h2, h1

    # Reconstruct
    def forward(self, x, label, grl_scale=0.1, testmode=False):
        if testmode:
            mu = self.encode(x, testmode=True)
            return self.decode(mu,label),self.discriminate(mu,grl_scale) # When the model is in test mode, the mu (without noise) is fed into the decoder
        else:
            z, mu, lnvar = self.encode(x, testmode=False)
            y=self.decode(z,label)
            dis_result = self.discriminate(mu,grl_scale)
            
            return y, mu, lnvar,dis_result



# The U-Net for the latent diffusion model
class LDDPM_UNet(nn.Module):


    # ZC: the number of channels in the latent feature map encoded by the VAE
    # time_embed_dim : the dimension of the code vector for encoding time step information (even number)
    # num_groups: the number of groups in Group Normalization
    def __init__(self, ZC, time_embed_dim, num_groups=16):
        super(LDDPM_UNet, self).__init__()

        # the number of channels in each layer
        L1_C = 320
        L2_C = 2*L1_C
        L3_C = 4*L1_C
        L4_C = 4*L1_C
  
        # The layer responsible for encoding time step information
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # The layer responsible for the first convolution applied to the input image
        self.init_conv = nn.Conv2d(in_channels=ZC, out_channels=L1_C, kernel_size=1, stride=1, padding=0)

        # The downsampling layers of the U-Net
        self.down1 = DDPMDownSamplingLayer(in_channels=L1_C, out_channels=L2_C, time_embed_dim=time_embed_dim, num_groups=num_groups)
        self.down2 = DDPMDownSamplingLayer(in_channels=L2_C, out_channels=L3_C, time_embed_dim=time_embed_dim, num_groups=num_groups)
        self.down3 = DDPMDownSamplingLayer(in_channels=L3_C, out_channels=L4_C, time_embed_dim=time_embed_dim, num_groups=num_groups, attention_type='linear')

        # The middle layer of the U-Net for the diffusion model
        self.mid = DDPMMiddleLayer(channels=L4_C, time_embed_dim=time_embed_dim, num_groups=num_groups, attention_type='linear')

        # The upsampling layers of the U-Net for the diffusion model
        self.up3 = DDPMUpSamplingLayer(in_channels=L4_C, out_channels=L3_C, time_embed_dim=time_embed_dim, num_groups=num_groups, attention_type='linear')
        self.up2 = DDPMUpSamplingLayer(in_channels=L3_C, out_channels=L2_C, time_embed_dim=time_embed_dim, num_groups=num_groups)
        self.up1 = DDPMUpSamplingLayer(in_channels=L2_C, out_channels=L1_C, time_embed_dim=time_embed_dim, num_groups=num_groups)

        # The last convolutional layer
        self.last_conv = PreNormConv2d(in_channels=L1_C, out_channels=ZC, num_groups=num_groups, kernel_size=1, stride=1, padding=0, init_scale=0.0)

    def forward(self, x, t):
        h = self.init_conv(x) 
        time_embedding = self.time_encoder(t) # encode the time step information into a code vector
        s1, h = self.down1(h, time_embedding) 
        s2, h = self.down2(h, time_embedding)
        s3, h = self.down3(h, time_embedding)
        h = self.mid(h, time_embedding) # The middle layer of the U-Net for the diffusion model
        h = self.up3(h, s3, time_embedding) 
        h = self.up2(h, s2, time_embedding)
        h = self.up1(h, s1, time_embedding)
        y = self.last_conv(h) # The last convolutional layer to produce the output
        return y
