import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Input is [batch, emb, time]
    block [ conv -> layer norm -> act -> dropout -> downsample]
    To do:
        add res blocks.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, p_conv_drop=0):
        super(ConvBlock, self).__init__()

        # use it instead stride.

        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                bias=False,
                                padding='same')

        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)

        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """
        - conv
        - norm
        - activation
        - downsample

        """

        x = self.conv1d(x)

        # norm by last axis.
        x = torch.transpose(x, -2, -1)
        x = self.norm(x)
        x = torch.transpose(x, -2, -1)

        x = self.activation(x)

        x = self.drop(x)

        x = self.downsample(x)

        return x


class UpConvBlock(nn.Module):
    def __init__(self, scale, **args):
        super(UpConvBlock, self).__init__()
        self.conv_block = ConvBlock(**args)
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)

    def forward(self, x):

        x = self.conv_block(x)
        x = self.upsample(x)
        return x


class HVATNet(nn.Module):
    def __init__(self,
                 n_electrodes=30,  # Число каналов
                 n_channels_out=21,  # Число каналов на выход (В нашем случае - 5 пальцев => 5 каналов
                 channels=[8, 16, 32, 32],  # Количество признаков на каждом слое кодировщика (decoder - симметрично)
                 kernel_sizes=[3, 3, 3],  # Размер ядра
                 strides=[4, 4, 4],  # Степень сжатия на каждом слое
                 dilation=[1, 1, 1],  # Коэффициент дилляции
                 decoder_reduce=1,   # Опциональный параметр уменьшения размерности декодировщика (1 - не уменьшаем)
                 ):

        super(HVATNet, self).__init__()

        self.n_electrodes = n_electrodes

        self.n_inp_features = n_electrodes
        self.n_channels_out = n_channels_out

        self.model_depth = len(channels)-1
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)  # Dimensionality reduction

        # create downsample blcoks in Sequentional manner.
        self.downsample_blocks = nn.ModuleList([ConvBlock(channels[i],
                                                          channels[i+1],
                                                          kernel_sizes[i],
                                                          stride=strides[i],
                                                          dilation=dilation[i]) for i in range(self.model_depth)])

        # TODO eto che za huinia
        channels = [ch//decoder_reduce for ch in channels[:-1]] + channels[-1:]  # channels

        module_list_ = [UpConvBlock(
            scale=strides[i],
            in_channels=channels[i+1] if i == self.model_depth-1 else channels[i+1]*2,
            out_channels=channels[i],
            kernel_size=kernel_sizes[i]
        ) for i in range(self.model_depth-1, -1, -1)]

        self.upsample_blocks = nn.ModuleList(module_list_)
        self.conv1x1_one = nn.Conv1d(channels[0]*2, self.n_channels_out, kernel_size=1, padding='same')

    def forward(self, x):
        batch, elec, time = x.shape
        # x = x.reshape(batch, -1, time)  # flatten the input
        x = self.spatial_reduce(x)

        skip_connection = []

        for i in range(self.model_depth):
            skip_connection.append(x)
            x = self.downsample_blocks[i](x)

        for i in range(self.model_depth):
            x = self.upsample_blocks[i](x)
            x = torch.cat((x, skip_connection[-1-i]),  # skip connections
                          dim=1)
        x = self.conv1x1_one(x)

        x = x.reshape(batch, -1, 4, time)

        phi_2, a1, a2, a3 = x[:, :, 0], x[:, :, 1],  x[:, :,  2], x[:, :, 3]
        sin, cos = torch.sin(phi_2), torch.cos(phi_2)

        qx, qy, qz = a1*sin, a2*sin, a3*sin
        qw = cos
        x = torch.stack([qx, qy, qz, qw], 2)

        # -1, 1 and normalization of the vectors

        # x = F.normalize(torch.tanh(x), 2)
        x = torch.clip(x, -1, 1)

        return x

    @torch.no_grad()
    def inference(self, myo, device='cpu', first_bone_is_constant=False):
        """
        Params:
            myo: is numpy array with shape (N_timestamps, 8)
        Return
            numpy array with shape (N_timestamps, N_bones, 4)
        """
        self.eval()
        self.to(device)

        x = torch.from_numpy(myo).T
        x = x.to(device).float()
        x = x.unsqueeze(0)

        y_pred = self(x)

        y_pred = y_pred[0].to('cpu').detach().numpy()
        y_pred = np.transpose(y_pred, (2, 0, 1))

        if first_bone_is_constant:
            y_pred[:, 0, :] = [0, 0, 0, -1]

        return y_pred

    
    
    
    

    
    
def freeze_all(model):
    for layer in model.modules():
        for param in layer.parameters():
            param.requires_grad = False
    return model 

def unfreeze_layer_norms(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.LayerNorm): 
            for param in layer.parameters():
                param.requires_grad = True
    return model

def unfreeze_first_layer(model):
    layers = []
    for i, child in enumerate(model.children()):
        layers.append(child)
        
    first_layer, encoder, decoder, last_layer = layers
    for param in first_layer.parameters():
        param.requires_grad = True
        
    return model