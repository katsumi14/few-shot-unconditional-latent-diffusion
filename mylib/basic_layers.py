import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


# The module for initializing the weights of the PixShuffle layer
def __ICNR__(tensor, initializer, upscale_factor=2, *args, **kwargs):
    upscale_factor_squared = upscale_factor * upscale_factor
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared, *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)

# The module for selecting the activation function
def __select_activation__(activation):
    activation = activation.lower()
    if activation == 'e' or activation == 'elu':
        act = F.elu
    elif activation == 'g' or activation == 'gelu':
        act = F.gelu
    elif activation == 'l' or activation == 'leaky-relu' or activation == 'leaky_relu':
        act = F.leaky_relu
    elif activation == 'p' or activation == 'prelu':
        act = F.prelu
    elif activation == 'r' or activation == 'relu':
        act = F.relu
    elif activation == 'w' or activation == 'silu' or activation == 'swish':
        act = F.silu
    elif activation == 's' or activation == 'sigmoid':
        act = torch.sigmoid
    elif activation == 't' or activation == 'tanh':
        act = torch.tanh
    else:
        act = None
    return act


# The module for adding normalization layer and activation function to the base layer
def __wrap_layer__(layer:nn.Module, normalization:str='none', activation:str='none', pre_act:bool=False, **kwargs):

    # crate activation function
    activation = activation.lower()
    if activation == 'e' or activation == 'elu':
        act = nn.ELU()
    elif activation == 'g' or activation == 'gelu':
        act = nn.GELU()
    elif activation == 'l' or activation == 'leaky-relu' or activation == 'leaky_relu':
        act = nn.LeakyReLU()
    elif activation == 'p' or activation == 'prelu':
        act = nn.PReLU()
    elif activation == 'r' or activation == 'relu':
        act = nn.ReLU()
    elif activation == 'w' or activation == 'silu' or activation == 'swish':
        act = nn.SiLU()
    elif activation == 's' or activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 't' or activation == 'tanh':
        act = nn.Tanh()
    else:
        act = None

    # create normalization layer
    normalization = normalization.lower()
    if normalization == 'b' or normalization == 'batch':
        if type(layer) == torch.nn.modules.conv.Conv3d or type(layer) == torch.nn.modules.conv.ConvTranspose3d:
            norm = nn.BatchNorm3d(num_features=kwargs['num_features'])
        elif type(layer) == torch.nn.modules.conv.Conv2d or type(layer) == torch.nn.modules.conv.ConvTranspose2d:
            norm = nn.BatchNorm2d(num_features=kwargs['num_features'])
        else:
            norm = nn.BatchNorm1d(num_features=kwargs['num_features'])
    elif normalization == 'g' or normalization == 'group':
        norm = nn.GroupNorm(num_channels=kwargs['num_features'], num_groups=kwargs['num_groups'])
    elif normalization == 'i' or normalization == 'instance':
        if type(layer) == torch.nn.modules.conv.Conv3d or type(layer) == torch.nn.modules.conv.ConvTranspose3d:
            norm = nn.InstanceNorm3d(num_features=kwargs['num_features'])
        elif type(layer) == torch.nn.modules.conv.Conv2d or type(layer) == torch.nn.modules.conv.ConvTranspose2d:
            norm = nn.InstanceNorm2d(num_features=kwargs['num_features'])
        elif type(layer) == torch.nn.modules.conv.Conv1d or type(layer) == torch.nn.modules.conv.ConvTranspose1d:
            norm = nn.InstanceNorm1d(num_features=kwargs['num_features'])
        else:
            norm = None
    elif normalization == 'l' or normalization == 'layer':
        norm = nn.LayerNorm(normalized_shape=kwargs['normalized_shape'])
    elif normalization == 's' or normalization == 'spectral':
        layer = nn.utils.spectral_norm(layer)
        norm = None
    else:
        norm = None

    # add the normalization layer and the activation function
    if act is None and norm is None:
        return layer
    modules = []
    if pre_act:
        # pre-activation 
        if norm is not None:
            modules.append(norm)
        if act is not None:
            modules.append(act)
        modules.append(layer)
    else:
        # post-activation 
        modules.append(layer)
        if norm is not None:
            modules.append(norm)
        if act is not None:
            modules.append(act)
    return nn.Sequential(*modules)


# conv + normalization + activation function layer
# -in_channels: the number of channels of the input feature map
# - out_channels: the number of channels of the output feature map
# -kernel_size: kernel size (default is 3)
# - stride: stride width (default is 1)
# - padding: padding size (default is (kernel_size-1)/2)
# -dropout_ratio: if not 0, dropout is applied with this ratio (default is 0, i.e. no dropout)
# -normalization: normalization method ('none', 'batch', 'spectral', 'layer', etc. Default is 'batch')
# -activation: activation function (default is ReLU)
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=-1, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(Conv, self).__init__()
        if type(padding) == int and padding < 0:
            padding = (kernel_size - 1) // 2
        self.dropout_ratio = dropout_ratio
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = __wrap_layer__(self.conv, normalization=normalization, activation=activation, num_features=out_channels, **kwargs) # 正規化層と活性化関数の付加
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.conv(x) # 畳込み + 正規化 + 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h

# Conv+ Normalization + Activation function layer
# The defalut kernel_size is 4
# We assume that the input feature map has even height and width, and the kernel_size is also even
class ConvHalf(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(ConvHalf, self).__init__()
        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size-1)//2,
            dropout_ratio=dropout_ratio,
            normalization=normalization,
            activation=activation,
            **kwargs)

    def __call__(self, x):
        return self.conv(x)


# 逆畳込み + 正規化 + 活性化関数を行う層
# inConv + Normalization + Activation function layer
# -in_channels: the number of channels of the input feature map
# -out_channels: the number of channels of the output feature map
# -kernel_size: kernel size (default is 4)
# -stride: stride width (default is 2)
# -padding: padding size (default is (kernel_size-1)/2)
# -dropout_ratio: if not 0, dropout is applied with this ratio (default is 0, i.e. no dropout)
# -normalization: normalization method ('none', 'batch', 'spectral', 'layer', etc. Default is 'batch')
# -activation: activation function (default is ReLU)
class Deconv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=-1, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(Deconv, self).__init__()
        if type(padding) == int and padding < 0:
            padding = (kernel_size - 1) // 2
        self.dropout_ratio = dropout_ratio
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.deconv = __wrap_layer__(self.deconv, normalization=normalization, activation=activation, num_features=out_channels, **kwargs) # 正規化層と活性化関数の付加
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.deconv(x) # 逆畳込み + 正規化 + 活性化関数
        if self.dropout_ratio != 0:
            h = self.dropout(h) # ドロップアウト
        return h



# Pooling layer
# -method: pooling method ('max' or 'avg', default is average pooling)
# -scal:e: kernel size and stride width
class Pool(nn.Module):

    def __init__(self, method='avg', scale=2):
        super(Pool, self).__init__()
        if method == 'max':
            self.pool = nn.MaxPool2d(kernel_size=scale, stride=scale)
        else:
            self.pool = nn.AvgPool2d(kernel_size=scale, stride=scale)

    def __call__(self, x):
        return self.pool(x)


# Global Pooling layer
# -method: pooling method ('max' or 'avg', default is global average pooling)
# -output_size: the size of the output map (height, width)
class GlobalPool(nn.Module):

    def __init__(self, method='avg', output_size=1):
        super(GlobalPool, self).__init__()
        if method == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def __call__(self, x):
        return self.pool(x)



# Conv + Normalization + Activation function + Pixel Shuffle layer
# -in_channels: the number of channels of the input feature map
# -out_channels: the number of channels of the output feature map
# -kernel_size: kernel size (default is 3)
# -dropout_ratio: if not 0, dropout is applied with this ratio (default is 0, i.e. no dropout)
# -normalization: normalization method ('none', 'batch', 'spectral', 'layer', etc. Default is 'batch')
# -activation: activation function (default is ReLU)
class PixShuffle(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(PixShuffle, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*(scale**2), kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv.weight.data.copy_(__ICNR__(self.conv.weight, initializer=nn.init.kaiming_normal_, upscale_factor=scale))
        self.conv = __wrap_layer__(self.conv, normalization=normalization, activation=activation, num_features=out_channels*(scale**2), **kwargs) # add activation function and normalization layer
        self.ps = nn.PixelShuffle(scale)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.conv(x) # Conv + normalization + activation function
        h = self.ps(h) # pixel shuffle
        if self.dropout_ratio != 0:
            h = self.dropout(h) # dropout
        return h


# Fully Connected + Normalization + Activation function layer
# -in_features: the number of input features
# -out_features: the number of output features
# -dropout_ratio: if not 0, dropout is applied with this ratio (default is 0, i.e. no dropout)
# -normalization: normalization method ('none', 'batch', 'spectral', 'layer', etc. Default is 'batch')
# -activation: activation function (default is ReLU)
class FC(nn.Module):

    def __init__(self, in_features, out_features, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(FC, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc = __wrap_layer__(self.fc, normalization=normalization, activation=activation, num_features=out_features, **kwargs) # add normalization layer and activation function
        if dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)

    def __call__(self, x):
        h = self.fc(x) # Fully Connected + Normalization + Activation function
        if self.dropout_ratio != 0:
            h = self.dropout(h) # dropout
        return h


# Flatten layer
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()
        self.flat = nn.Flatten()

    def __call__(self, x):
        return self.flat(x)


# reverse of Flatten layer
# -size: the size of the reconfigured feature map
class Reshape(nn.Module):

    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def __call__(self, x):
        batchsize = x.size()[0]
        h = x.reshape((batchsize, *self.size))
        return h


# Conditional Batch Normalization
class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self.gamma_embed = nn.Linear(num_classes, num_features, bias=False)
        self.beta_embed = nn.Linear(num_classes, num_features, bias=False)

    def __call__(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_embed(y) + 1
        beta = self.beta_embed(y)
        return gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)


# Adaptive Instance Normalization
#   - x: contents feature map
#   - y: style feature map
class AdaIN(nn.Module):

    def __init__(self, in_channels_content, in_channels_style):
        super(AdaIN, self).__init__()
        if in_channels_content == in_channels_style:
            self.conv = None
        else:
            self.conv = Conv(in_channels=in_channels_style, out_channels=in_channels_content, kernel_size=1, stride=1, padding=0, normalization='none', activation='none')

    def __call__(self, x, y):
        if self.conv is not None:
            y = self.conv(y)
        xu = torch.mean(x, dim=(2, 3), keepdim=True) # average of the content feature map for each channel
        yu = torch.mean(y, dim=(2, 3), keepdim=True) # average of the style feature map for each channel
        xs = torch.std(x, dim=(2, 3), unbiased=False, keepdim=True) # standard deviation of the content feature map for each channel
        ys = torch.std(y, dim=(2, 3), unbiased=False, keepdim=True) # standard deviation of the style feature map for each channel
        h = ys * ((x - xu) / xs) + yu
        return h


# pre-processing layer for using VGG or ResNet as the backbone model for transfer learning
class BackbonePreprocess(nn.Module):

    def __init__(self, image_size=224, image_size_before_cropped=256, do_center_crop=True):
        super(BackbonePreprocess, self).__init__()
        if do_center_crop:
            self.preprocess = transforms.Compose([
                transforms.Resize(image_size_before_cropped, antialias=True), # normalize the 256x256 pixels
                transforms.CenterCrop(image_size), # extract the central 224x224 pixels from the 256x256 pixel input image
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize the pixel values of the extracted part
            ])
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize(image_size, antialias=True), # normalize the input image to 224x224 pixels
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize the pixel values of the extracted part
            ])

    def __call__(self, x):
        return self.preprocess(x)


# backbone layer for transfer learning
#   - model: the backbone model (e.g. VGG or ResNet) to be used for transfer learning
#   - layer_name: the name of the layer to be used as a feature 
#   - finetune: whether to fine-tune the backbone model (True for fine-tuning, False for transfer learning only. Default is False)
class Backbone(nn.Module):

    def __init__(self, model, layer_name, finetune=False):
        super(Backbone, self).__init__()
        self.finetune = finetune
        self.backbone = create_feature_extractor(model, {layer_name: 'feature'})
        if not finetune:
            # fix the parameters of the backbone model and set it to evaluation mode
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def __call__(self, x):
        return self.backbone(x)['feature']

    def train(self, mode: bool = True):
        if self.finetune:
            super().train(mode)
            self.backbone.train(mode)
        else:
            super().train(False)
            self.backbone.train(False)
        return self

    def eval(self):
        return self.train(False)

    def print_output_size(self):
        for param in self.parameters():
            device = param.data.device
            break
        x = torch.randn(1, 3, 224, 224).to(device)
        y = self.__call__(x)
        print(y.size()[1:])

    def get_output_size(self):
        for param in self.parameters():
            device = param.data.device
            break
        x = torch.randn(1, 3, 224, 224).to(device)
        y = self.__call__(x)
        return y.size()[1:]


# Residual Block layer of the plain type
# stride and padding in convolution are automatically determined so that the output map has the same size as the input map
#   - in_channels: the number of channels in the input map
#   - out_channels: the number of channels in the output map (if 0 or negative, it will be automatically set to in_channels, default is 0)
#   - kernel_size: the size of the kernel (only odd numbers are allowed, even numbers may not work as expected. default is 3)
#   - dropout_ratio: if not 0, apply dropout with the specified ratio (default is 0, i.e., no dropout)
#   - normalization: the normalization method ('none', 'batch', 'spectral', 'layer', etc. default is 'batch')
#   - activation: the activation function (default is ReLU)
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels=0, kernel_size=3, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(ResBlock, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        if out_channels <= 0:
            out_channels = in_channels
        self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,
                          normalization=normalization, activation=activation, **kwargs)
        self.conv2 = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,
                          normalization=normalization, activation='none', **kwargs)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        if in_channels == out_channels:
            self.shortcut = None
        else:
            normalization = normalization.lower()
            if normalization == 's' or normalization == 'spectral':
                self.shortcut = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, normalization='spectral', activation='none')
            else:
                self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h = self.conv1(x) # Conv + Normalization + Activation
        h = self.conv2(h) # Conv + Normalization
        if self.shortcut is not None:
            x = self.shortcut(x)
        h = h + x # add shortcut
        if self.activation is not None:
            h = self.activation(h) # activation function
        if self.dropout_ratio != 0:
            h = self.dropout(h) # dropout
        return h


# The Residual Block layer of the pre-activation type
# straide and padding in convolution are automatically determined so that the output map has the same size as the input map
#   - in_channels: the number of channels in the input map
#   - out_channels: the number of channels in the output map (if 0 or negative, it will be automatically set to in_channels, default is 0)
#   - kernel_size: the size of the kernel (only odd numbers are allowed, even numbers may not work as expected. default is 3)
#   - dropout_ratio: if not 0, apply dropout with the specified ratio (default is 0, i.e., no dropout)
#   - normalization: the normalization method ('none', 'batch', 'spectral', 'layer', etc. default is 'batch')
#   - activation: the activation function (default is ReLU)
class ResBlockPA(nn.Module):

    def __init__(self, in_channels, out_channels=0, kernel_size=3, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(ResBlockPA, self).__init__()
        self.dropout_ratio = dropout_ratio
        if out_channels <= 0:
            out_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv1 = __wrap_layer__(self.conv1, normalization=normalization, activation=activation, pre_act=True, num_features=in_channels, **kwargs) # add normalization layer and activation function
        self.conv2 = __wrap_layer__(self.conv2, normalization=normalization, activation=activation, pre_act=True, num_features=out_channels, **kwargs) # add normalization layer and activation function
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        if in_channels == out_channels:
            self.shortcut = None
        else:
            normalization = normalization.lower()
            if normalization == 's' or normalization == 'spectral':
                self.shortcut = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, normalization='spectral', activation='none')
            else:
                self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h = self.conv1(x) # Normalization + Activation function + Conv
        h = self.conv2(h) # Normalization + Activation function + Conv
        if self.shortcut is not None:
            x = self.shortcut(x)
        h = h + x # add shortcut
        if self.dropout_ratio != 0:
            h = self.dropout(h) # dropout
        return h


# The Residual Block layer of the pre-activation type
# stride and padding in convolution are automatically determined so that the output map has the same size as the input map
#   -in_channels: the number of channels in the input map
#   -out_channels: the number of channels in the output map (if 0 or negative, it will be automatically set to in_channels, default is 0)
#   - mid_channels: the number of channels in the intermediate map (if 0 or negative, it will be automatically set to in_channels // 4, default is 0)
#   - kernel_size: the size of the kernel (only odd numbers are allowed. default is 3)
#   - dropout_ratio: if not 0, apply dropout with the specified ratio (default is 0, i.e., no dropout)
#   - normalization: normalization method ('none', 'batch', 'spectral', 'layer' etc. Default is 'batch')
#   - activation: activation function (Default is ReLU)
class ResBlockBN(nn.Module):

    def __init__(self, in_channels, out_channels=0, mid_channels=0, kernel_size=3, dropout_ratio=0, normalization='batch', activation='relu', **kwargs):
        super(ResBlockBN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.activation = __select_activation__(activation)
        if out_channels <= 0:
            out_channels = in_channels
        if mid_channels <= 0:
            mid_channels = in_channels // 4
        self.conv1 = Conv(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0,
                          normalization=normalization, activation=activation, **kwargs)
        self.conv2 = Conv(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,
                          normalization=normalization, activation=activation, **kwargs)
        self.conv3 = Conv(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0,
                          normalization=normalization, activation='none', **kwargs)
        if dropout_ratio != 0:
            self.dropout = nn.Dropout2d(p=self.dropout_ratio)
        if in_channels == out_channels:
            self.shortcut = None
        else:
            normalization = normalization.lower()
            if normalization == 's' or normalization == 'spectral':
                self.shortcut = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, normalization='spectral', activation='none')
            else:
                self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        h = self.conv1(x) # convolution + normalization + activation function
        h = self.conv2(h) # convolution + normalization + activation function
        h = self.conv3(h) # convolution + normalization
        if self.shortcut is not None:
            x = self.shortcut(x)
        h = h + x # add shortcut
        if self.activation is not None:
            h = self.activation(h) # activation function
        if self.dropout_ratio != 0:
            h = self.dropout(h) # dropout
        return h


#Minibatch Standard Deviation layer for GANs
class MinibatchStdev(nn.Module):

    def __init__(self):
        super(MinibatchStdev, self).__init__()

    def __call__(self, h):
        size = h.size()
        size = (size[0], 1, size[2], size[3])
        return torch.cat((h, torch.mean(torch.std(h, dim=0, unbiased=False)).repeat(size)), dim=1)


#Minibatch Discrimination layer for GANs
# citation: https://gist.github.com/t-ae/732f78671643de97bbe2c46519972491
class MinibatchDiscrimination(nn.Module):

    def __init__(self, in_features, out_features, kernel_dims=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def __call__(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)
        M = matrices.unsqueeze(0) # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3) # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3) # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1) # NxB, subtract self distance
        return torch.cat([x, o_b], 1)


# the class of Discriminator Augmentation for GANs
class DiscriminatorAugmentation(nn.Module):

    # H: height of the input image
    # W: width of the input image
    # p_hflip: probability of horizontal flip
    # p_vflip: probability of vertical flip (only for non-square images)
    # p_rot: probability of rotation (90, 180, 270 degrees) (only for square images)
    def __init__(self, H, W, p_hflip, p_vflip, p_rot):
        super(DiscriminatorAugmentation, self).__init__()
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot = p_rot
        if H == W:
            self.is_square = True
        else:
            self.is_square = False

    def forward(self, *args):
        ret = args
        if torch.rand(1) < self.p_hflip:
            ret = [transforms.functional.hflip(x) for x in args]
        if self.is_square:
            if torch.rand(1) < self.p_rot:
                angle = int(90 * (torch.randint(3, (1,)) + 1))
                ret = [transforms.functional.rotate(x, angle, fill=0) for x in ret]
        else:
            if torch.rand(1) < self.p_vflip:
                ret = [transforms.functional.vflip(x) for x in ret]
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


# Cosine-similarity-based recognition layer
# cite:https://github.com/MuggleWang/CosFace_pytorch
class CosineSimScore(nn.Module):

    # calculate cosine similarity
    @staticmethod
    def cosine_sim(x1, x2, dim=1, eps=1e-8):
        ip = torch.mm(x1, x2.t()) # the inner product of x1 and x2
        w1 = torch.norm(x1, 2, dim) # the squared norm of x1
        w2 = torch.norm(x2, 2, dim) # the squared norm of x2
        return ip / torch.outer(w1, w2).clamp(min=eps)

    # constructor
    #   - in_features: the number of units on the input side (dimension of features)
    #   - out_features: the number of units on the output side (number of recognition targets)
    #   - margin: the margin width added to the cosine similarity of the correct class
    #   - scale: the amplification rate of the cosine similarity
    def __init__(self, in_features, out_features, margin=0.35, scale=30.0):
        super(CosineSimScore, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    # forward pass (when using this layer, ground truth label information is required during training. Not needed during testing)
    #   - x: input image (mini-batch)
    #   - t: ground truth label (mini-batch, only needed during training)
    def forward(self, x, t=None):
        cosine = self.cosine_sim(x, self.weight)
        if t is None:
            return self.s * cosine
        else:
            # ground truth label is given as an integer value, convert it to one-hot representation
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, t.view(-1, 1), 1.0)
            return self.s * (cosine - one_hot * self.m)
