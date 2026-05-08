import math
import torch
import torch.nn as nn
from mylib.basic_layers import ConditionalBatchNorm2d



# The layer that computes the time step embedding vector
# - time_embed_dim: the dimension of the time step embedding vector
class SinusoidalTimeEmbeddings(nn.Module):

    def __init__(self, time_embed_dim):
        super(SinusoidalTimeEmbeddings, self).__init__()
        self.embed_dim = time_embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        embeddings = torch.log(torch.tensor(10000, device=t.device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# The layer that applies Group Normalization + Siwsh before convolution
# - num_groups: the number of groups in Group Normalization
class PreNormConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups, kernel_size, stride, padding, normalize=True, init_scale=1.0):
        super(PreNormConv2d, self).__init__()
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels) if normalize else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        nn.init.xavier_uniform_(self.conv.weight, gain=math.sqrt(init_scale or 1e-10))
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        h = self.norm(x)
        h = self.act(h)
        return self.conv(h)


# Resblock that considers time step information
#  -num_groups: the number of groups in Group Normalization
#  -time_embed_dim: the dimension of the time step embedding vector (if <=0, it becomes a normal ResBlock)
class DDPMResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups, kernel_size=3, time_embed_dim=0):
        super(DDPMResBlock, self).__init__()
        if time_embed_dim > 0:
            self.mlp = nn.Sequential(
                nn.SiLU(), 
                nn.Linear(time_embed_dim, out_channels),
            )
        else:
            self.mlp = None
        self.block1 = PreNormConv2d(in_channels, out_channels, num_groups=num_groups, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.block2 = PreNormConv2d(out_channels, out_channels, num_groups=num_groups, kernel_size=kernel_size, stride=1, padding=kernel_size//2, init_scale=0.0)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time_embedding=None):
        h = self.block1(x)
        if self.mlp is not None:
            h = h + self.mlp(time_embedding).unsqueeze(2).unsqueeze(3)
        h = self.block2(h)
        return h + self.skip(x)

# ResBlock that considers label information
# -num_groups: the number of groups in Group Normalization
# -label_embed_dim: the dimension of the label embedding vector (if <=0, it becomes a normal ResBlock)
class DDPMResBlock_Alt(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups, kernel_size=3, label_embed_dim=0):
        super(DDPMResBlock_Alt, self).__init__()
        if label_embed_dim > 0:
            self.mlp = nn.Sequential(
                nn.SiLU(), 
                nn.Linear(label_embed_dim, out_channels),
            )
        else:
            self.mlp = None
        self.block1 = PreNormConv2d(in_channels, out_channels, num_groups=num_groups, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.block2 = PreNormConv2d(out_channels, out_channels, num_groups=num_groups, kernel_size=kernel_size, stride=1, padding=kernel_size//2, init_scale=0.0)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, label_embedding=None):
        h = self.block1(x)
        if self.mlp is not None:
            h = h + self.mlp(label_embedding).unsqueeze(2).unsqueeze(3)
        h = self.block2(h)
        return h + self.skip(x)


# Linear Attention
# -num_groups: the number of groups in Group Normalization
# -num_heads: the number of heads in multi-head attention
# -embed_dim: the dimension of each head (different from the dimension of the time step embedding vector)
class DDPMLinearAttention(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups, num_heads, embed_dim):
        super(DDPMLinearAttention, self).__init__()
        self.scale = embed_dim ** (- 0.5)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.to_qkv = nn.Conv2d(in_channels, num_heads * embed_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(num_heads * embed_dim, out_channels, 1)
        nn.init.xavier_uniform_(self.to_out.weight, gain=1e-5)

    # x: feature map (batch_size, channel_num, height, width)
    def forward(self, x):
        B, _, H, W = x.size()
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q = torch.reshape(q, (B, self.num_heads, self.embed_dim, H * W))
        k = torch.reshape(k, (B, self.num_heads, self.embed_dim, H * W))
        v = torch.reshape(v, (B, self.num_heads, self.embed_dim, H * W))
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v).contiguous()
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q).contiguous()
        out = torch.reshape(out, (B, self.num_heads * self.embed_dim, H, W))
        return self.to_out(out) + self.skip(x)


# General Multi-head Attention
# -num_groups: the number of groups in Group Normalization
# -num_heads: the number of heads in multi-head attention
# -embed_dim: the dimension of each head (different from the dimension of the time step embedding vector)
class DDPMAttention(nn.Module):

    def __init__(self, in_channels, out_channels, num_groups, num_heads, embed_dim):
        super(DDPMAttention, self).__init__()
        self.scale = embed_dim ** (- 0.5)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.to_qkv = nn.Conv2d(in_channels, num_heads * embed_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(num_heads * embed_dim, out_channels, 1)
        nn.init.xavier_uniform_(self.to_out.weight, gain=1e-5)

    # x: feature map (batch_size, channel_num, height, width)
    def forward(self, x):
        B, _, H, W = x.size()
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q = torch.reshape(q, (B, self.num_heads, self.embed_dim, H * W))
        k = torch.reshape(k, (B, self.num_heads, self.embed_dim, H * W))
        v = torch.reshape(v, (B, self.num_heads, self.embed_dim, H * W))
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k).contiguous()
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v).contiguous()
        out = torch.reshape(out.permute(0, 1, 3, 2), (B, self.num_heads * self.embed_dim, H, W))
        return self.to_out(out) + self.skip(x)


# The function for selecting the type of attention
def get_attention_block(attention_type, in_channels, out_channels, num_groups, num_heads, embed_dim):
    if attention_type == 'linear':
        attn = DDPMLinearAttention(in_channels=in_channels, out_channels=out_channels, num_groups=num_groups, num_heads=num_heads, embed_dim=embed_dim)
    elif attention_type == 'normal':
        attn = DDPMAttention(in_channels=in_channels, out_channels=out_channels, num_groups=num_groups, num_heads=num_heads, embed_dim=embed_dim)
    elif attention_type == 'none':
        attn = nn.Identity()
    else:
        raise NotImplementedError()
    return attn


# The middle layer of the U-Net for the diffusion model
# -time_embed_dim: the dimension of the time step embedding vector
# -num_groups: the number of groups in Group Normalization
# -num_heads: the number of heads in multi-head attention (the dimension of each head is specified by channels/num_heads)
# -attention_type: if 'normal', normal multi-head attention is used; if 'linear', linear attention is used; if 'none', no attention is used
class DDPMMiddleLayer(nn.Module):

    def __init__(self, channels, time_embed_dim, num_groups, num_heads=8, attention_type='none'):
        super(DDPMMiddleLayer, self).__init__()
        embed_dim = channels // num_heads
        self.block1 = DDPMResBlock(in_channels=channels, out_channels=channels, num_groups=num_groups, kernel_size=3, time_embed_dim=time_embed_dim)
        self.block2 = DDPMResBlock(in_channels=channels, out_channels=channels, num_groups=num_groups, kernel_size=3, time_embed_dim=time_embed_dim)
        self.attn = get_attention_block(attention_type, channels, channels, num_groups, num_heads, embed_dim)

    def forward(self, x, time_embedding=None):
        h = self.block1(x, time_embedding)
        h = self.attn(h)
        y = self.block2(h, time_embedding)
        return y

# The downsampling layer of the U-Net and VAE for the diffusion model
# -time_embed_dim: the dimension of the time step embedding vector
# -num_groups: the number of groups in Group Normalization
# -num_heads: the number of heads in multi-head attention (the dimension of each head is specified by channels/num_heads)
# -attention_type: if 'normal', normal multi-head attention is used; if 'linear', linear attention is used; if 'none', no attention is used
# -with_downsample: if False, no downsampling is performed
# -with_skip_output: if False, no features for skip connections are output
class DDPMDownSamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels, time_embed_dim, num_groups, num_heads=8, attention_type='none', with_downsample=True, with_skip_output=True):
        super(DDPMDownSamplingLayer, self).__init__()
        embed_dim = in_channels // num_heads
        self.with_skip_output = with_skip_output
        self.block1 = DDPMResBlock(in_channels=in_channels, out_channels=in_channels, num_groups=num_groups, kernel_size=3, time_embed_dim=time_embed_dim)
        self.block2 = DDPMResBlock(in_channels=in_channels, out_channels=in_channels, num_groups=num_groups, kernel_size=3, time_embed_dim=time_embed_dim)
        self.attn = get_attention_block(attention_type, in_channels, in_channels, num_groups, num_heads, embed_dim)
        if with_downsample:
            self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.down = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time_embedding=None):
        h = self.block1(x, time_embedding)
        s = self.attn(h) # use the output of this block as a skip connection to the upsampling layer
        h = self.block2(s, time_embedding)
        y = self.down(h)
        if self.with_skip_output:
            return s, y
        else:
            return y


# The upsampling layer of VAE for the diffusion model
#   - time_embed_dim: the dimension of the time step embedding vector
#   - num_groups: the number of groups in Group Normalization
#   - num_heads: the number of heads in multi-head attention (the dimension of each head is specified by out_channels/num_heads)
#   - attention_type: if 'normal', normal multi-head attention is used; if 'linear', linear attention is used; if 'none', no attention is used
#   - with_upsample: if False, no upsampling is performed
#   - with_skip_input: if False, no features for skip connections are accepted
class DDPMUpSamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels, time_embed_dim, num_groups, num_heads=8, attention_type='none', with_upsample=True, with_skip_input=True):
        super(DDPMUpSamplingLayer, self).__init__()
        embed_dim = out_channels // num_heads
        block1_out_channels = out_channels * 2 if with_skip_input else out_channels
        self.block1 = DDPMResBlock(in_channels=block1_out_channels, out_channels=out_channels, num_groups=num_groups, kernel_size=3, time_embed_dim=time_embed_dim)
        self.block2 = DDPMResBlock(in_channels=out_channels, out_channels=out_channels, num_groups=num_groups, kernel_size=3, time_embed_dim=time_embed_dim)
        self.attn = get_attention_block(attention_type, out_channels, out_channels, num_groups, num_heads, embed_dim)
        if with_upsample:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, s=None, time_embedding=None):
        h = self.up(x)
        if s is not None:
            h = torch.cat((h, s), dim=1)
        h = self.block1(h, time_embedding)
        h = self.attn(h)
        y = self.block2(h, time_embedding)
        return y
        



# The upsampling layer of the U-Net for the diffusion model
#   - time_embed_dim: the dimension of the time step embedding vector
#   - num_groups: the number of groups in Group Normalization
#   - num_heads: the number of heads in multi-head attention (the dimension of each head is specified by out_channels/num_heads)
#   - attention_type: if 'normal', normal multi-head attention is used; if 'linear', linear attention is used; if 'none', no attention is used
#   - with_upsample: if False, no upsampling is performed
#   - with_skip_input: if False, no features for skip connections are accepted
class DDPMUpSamplingLayer_Alt(nn.Module):

    def __init__(self, in_channels, out_channels, label_embed_dim, num_groups, num_heads=8, attention_type='none', with_upsample=True, with_skip_input=True):
        super(DDPMUpSamplingLayer_Alt, self).__init__()
        embed_dim = out_channels // num_heads
        block1_out_channels = out_channels * 2 if with_skip_input else out_channels
        self.block1 = DDPMResBlock_Alt(in_channels=block1_out_channels, out_channels=out_channels, num_groups=num_groups, kernel_size=3, label_embed_dim=label_embed_dim)
        self.block2 = DDPMResBlock_Alt(in_channels=out_channels, out_channels=out_channels, num_groups=num_groups, kernel_size=3, label_embed_dim=label_embed_dim)
        self.attn = get_attention_block(attention_type, out_channels, out_channels, num_groups, num_heads, embed_dim)
        if with_upsample:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, s=None, label_embedding=None):
        h = self.up(x)
        if s is not None:
            h = torch.cat((h, s), dim=1)
        h = self.block1(h, label_embedding)
        h = self.attn(h)
        y = self.block2(h, label_embedding)
        return y



