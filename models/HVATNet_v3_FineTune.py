import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary

import sys, os

# setting path
# sys.path.insert(1, os.path.realpath(os.path.pardir))
sys.path.append("../utils")


# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint


# 1. Input data -> Conv1d 1x1 -> N Res Blocks. -> M downsample blocks with advanced Conv -> M upsample blocks with skip connections +
# outputs -> Merge Multiscale features
from einops import rearrange

# from SALUT_ML.utils.quats_and_angles import get_quats
from utils.quats_and_angles import get_quats


class TuneModule(nn.Module):
    def __init__(self, n_electrodes=8, temperture=5):
        super(TuneModule, self).__init__()
        """
        - interpolate signal spatially 
        - change amplitude of the signal

        n_electrodes: number of electrodes (default: 8)
        temperture: temperature for softmax of weights (default: 5)
        """
        # spatial rotation.
        self.spatial_weights = torch.nn.Parameter(
            torch.eye(n_electrodes, n_electrodes), requires_grad=True
        )
        self.temp = torch.tensor(temperture, requires_grad=False)

        # normalization + amplitude scaling
        self.layer_norm = nn.LayerNorm(n_electrodes, elementwise_affine=True, eps=1e-5)

    def forward(self, x):
        """
        x: batch, channel, time
        """

        x = x.permute(0, 2, 1)  # batch, time, channel

        # spatial rotation
        weights = torch.softmax(self.spatial_weights * self.temp, dim=0)
        x = torch.matmul(x, weights)  # batch, time, channel

        # normalization + amplitude scaling
        x = self.layer_norm(x)

        x = x.permute(0, 2, 1)  # batch, channel, time

        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # get number of parameters
        self.n_params = 0
        for p in self.parameters():
            self.n_params += p.numel()
        print("Number of parameters: ", self.n_params)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out


class SimpleResBlock(nn.Module):
    """
    Input is [batch, emb, time]
    Res block.
    In features input and output the same.
    So we can apply this block several times.
    """

    def __init__(self, in_channels, kernel_size):
        super(SimpleResBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, bias=True, padding="same"
        )

        self.activation = nn.GELU()

        self.conv2 = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, bias=True, padding="same"
        )

    def forward(self, x_input):

        x = self.conv1(x_input)
        x = self.activation(x)
        x = self.conv2(x)

        res = x + x_input

        return res


class AdvancedConvBlock(nn.Module):
    """
    Input is [batch, emb, time]
    block [ conv -> layer norm -> act -> dropout ]

    To do:
        add res blocks.
    """

    def __init__(self, in_channels, kernel_size, dilation=1):
        super(AdvancedConvBlock, self).__init__()

        # use it instead stride.

        self.conv_dilated = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
            padding="same",
        )

        self.conv1_1 = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, bias=True, padding="same"
        )

        self.conv1_2 = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, bias=True, padding="same"
        )

        self.conv_final = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, bias=True, padding="same"
        )

    def forward(self, x_input):
        """
        input
            - dilation
            - gated convolution
            - conv final
            - maybe dropout. and LN
        - input + res
        """
        x = self.conv_dilated(x_input)

        flow = torch.tanh(self.conv1_1(x))
        gate = torch.sigmoid(self.conv1_2(x))
        res = flow * gate

        res = self.conv_final(res)

        res = res + x_input
        return res


class AdvancedEncoder(nn.Module):
    def __init__(
        self,
        n_blocks_per_layer=3,
        n_filters=64,
        kernel_size=3,
        dilation=1,
        strides=(2, 2, 2),
    ):
        super(AdvancedEncoder, self).__init__()

        self.n_layers = len(strides)
        self.downsample_blocks = nn.ModuleList(
            [
                nn.Conv1d(n_filters, n_filters, kernel_size=stride, stride=stride)
                for stride in strides
            ]
        )

        conv_layers = []
        for i in range(self.n_layers):
            blocks = nn.ModuleList(
                [
                    AdvancedConvBlock(n_filters, kernel_size, dilation=dilation)
                    for i in range(n_blocks_per_layer)
                ]
            )

            layer = nn.Sequential(*blocks)
            conv_layers.append(layer)

        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        """
        Apply conv + downamsple
        Return uutputs of eahc conv + the last features after downsampling.
        """

        outputs = []
        for conv_block, down in zip(self.conv_layers, self.downsample_blocks):
            x_res = conv_block(x)
            x = down(x_res)
            outputs.append(x_res)

        outputs.append(x)

        return outputs


class AdvancedDecoder(nn.Module):
    def __init__(
        self,
        n_blocks_per_layer=3,
        n_filters=64,
        kernel_size=3,
        dilation=1,
        strides=(2, 2, 2),
    ):
        super(AdvancedDecoder, self).__init__()

        self.n_layers = len(strides)

        self.upsample_blocks = nn.ModuleList(
            [
                nn.Upsample(scale_factor=scale, mode="linear", align_corners=False)
                for scale in strides
            ]
        )

        conv_layers = []
        for i in range(self.n_layers):

            reduce = nn.Conv1d(
                n_filters * 2, n_filters, kernel_size=kernel_size, padding="same"
            )
            conv_blocks = nn.ModuleList(
                [
                    AdvancedConvBlock(n_filters, kernel_size, dilation=dilation)
                    for i in range(n_blocks_per_layer)
                ]
            )

            conv_blocks.insert(0, reduce)
            layer = nn.Sequential(*conv_blocks)

            conv_layers.append(layer)

        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, skips):
        """
        Apply conv + downamsple
        Return uutputs of each conv + the last features after downsampling.
        """
        skips = skips[::-1]
        x = skips[0]

        outputs = []
        for idx, (conv_block, up) in enumerate(
            zip(self.conv_layers, self.upsample_blocks)
        ):
            x = up(x)

            x = torch.cat([x, skips[idx + 1]], 1)
            x = conv_block(x)

            outputs.append(x)

        return outputs


class HVATNetv3(nn.Module):
    def __init__(
        self,
        n_electrodes=8,
        n_channels_out=64,
        n_res_blocks=2,
        n_blocks_per_layer=2,
        n_filters=128,
        kernel_size=5,
        strides=(2, 2, 2, 2),
        dilation=2,
        small_strides=(2, 2),
        use_angles=False,
    ):

        super(HVATNetv3, self).__init__()

        # configuration of model
        self.n_inp_features = n_electrodes
        self.n_channels_out = n_channels_out
        self.model_depth = len(strides)
        self.use_angles = use_angles

        self.tune_module = TuneModule(n_electrodes=n_electrodes, temperture=5.0)

        # change number of features to custom one
        self.spatial_reduce = nn.Conv1d(
            n_electrodes, n_filters, kernel_size=1, padding="same"
        )

        self.denoiser = nn.Sequential(
            *[SimpleResBlock(n_filters, kernel_size) for i in range(n_res_blocks)]
        )

        self.encoder = AdvancedEncoder(
            n_blocks_per_layer=n_blocks_per_layer,
            n_filters=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            strides=strides,
        )

        self.mapper = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, kernel_size, padding="same"),
            nn.GELU(),
            nn.Conv1d(n_filters, n_filters, kernel_size, padding="same"),
            nn.GELU(),
        )

        self.encoder_small = AdvancedEncoder(
            n_blocks_per_layer=n_blocks_per_layer,
            n_filters=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            strides=small_strides,
        )
        self.decoder_small = AdvancedDecoder(
            n_blocks_per_layer=n_blocks_per_layer,
            n_filters=n_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            strides=small_strides[::-1],
        )

        # self.rnn = RNN(input_size= n_filters, hidden_size=n_filters, num_layers=5,output_size=n_filters)
        self.simple_pred_head = nn.Conv1d(
            n_filters, n_channels_out, kernel_size=1, padding="same"
        )

        # get number of parameters
        self.n_params = 0
        for p in self.parameters():
            self.n_params += p.numel()
        print("Number of parameters: ", self.n_params)

    def forward(self, x):
        """
        x: [batch, elec, time]
        """

        batch, elec, time = x.shape

        # tune inputs to model
        x = self.tune_module(x)

        # denoising part
        x = self.spatial_reduce(x)
        x = self.denoiser(x)

        # extract features
        # TODO: add mapper and change encoder to return all features
        outputs = self.encoder(x)
        emg_features = outputs[-1]  # 25 fps features

        # decode features
        # 1. simple way:  mapper + pred_head + quat conversion
        # emg_features = self.mapper(emg_features)

        # 2. Unet way:  encoder + mapper + decoder + quat conversion
        outputs_small = self.encoder_small(emg_features)
        outputs_small[-1] = self.mapper(outputs_small[-1])
        emg_features = self.decoder_small(outputs_small)[-1]

        # 3. RNN way
        # emg_features  = emg_features.permute(0, 2, 1) # size [batch, n_filters, time]
        # res = self.rnn(emg_features)
        # emg_features = res.permute(0, 2, 1)

        # quats conversion
        pred = self.simple_pred_head(emg_features)

        if self.use_angles:
            return pred
        else:
            pred_quat_shape = self._convert2quats(pred)
            return pred_quat_shape

    def _convert2quats(self, x):
        batch, n_outs, time = x.shape
        x = x.reshape(batch, -1, 4, time)

        if self.training:
            return x
        else:
            return F.normalize(x, p=2.0, dim=2)

    @torch.no_grad()
    def inference(self, myo, device="cpu", first_bone_is_constant=False):
        """
        Params:
            myo: is numpy array with shape (N_timestamps, 8)
        Return
            numpy array with shape (N_timestamps, N_bones, 4)
        """
        self.eval()
        self.to(device)

        x = torch.from_numpy(myo).T
        x = x.unsqueeze(0)
        x = x.to(device).float()

        y_pred = self(x)

        y_pred = y_pred[0].to("cpu").detach().numpy()

        if self.use_angles:
            y_pred = rearrange(y_pred, "a t -> t a")
            y_pred = get_quats(y_pred)
        else:
            y_pred = rearrange(y_pred, "b q t -> t b q")

        if first_bone_is_constant:
            y_pred[:, 0, :] = [0, 0, 0, -1]

        return y_pred

    @torch.no_grad()
    def inference_v2(self, myo, device="cpu", first_bone_is_constant=False):
        """
        Params:
            myo: is numpy array with shape (N_timestamps, 8)
        Return
            numpy array with shape (N_timestamps, 20)
        """
        self.eval()
        self.to(device)

        x = torch.from_numpy(myo).T
        x = x.unsqueeze(0)
        x = x.to(device).float()

        y_pred = self(x)

        y_pred = y_pred[0].to("cpu").detach().numpy()

        if self.use_angles:
            y_pred = rearrange(y_pred, "a t -> t a")
            # y_pred = get_quats(y_pred)
        else:
            y_pred = rearrange(y_pred, "b q t -> t b q")

        # if first_bone_is_constant:
        #     y_pred[:, 0, :] = [0, 0, 0, -1]

        return y_pred


# start python code
if __name__ == "__main__":

    hvatnet_v3_params = dict(
        n_electrodes=8,
        n_channels_out=64,
        n_res_blocks=3,
        n_blocks_per_layer=3,
        n_filters=128,
        kernel_size=3,
        strides=(2, 2, 2),
        dilation=2,
        small_strides=(2, 2),
    )
    model = HVATNetv3(**hvatnet_v3_params)

    x = torch.randn(1, 8, 256)
    y = model(x)

    print(summary(model.to("cpu"), x, show_input=True))
    print("Input shape: ", x.shape)
    print("Output shape: ", y.shape)
