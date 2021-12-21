import torch
import torch.nn as nn
from perceiver_pytorch import Perceiver


class AggregatorPerceiver(nn.Module):
    """
    They say it's all we need ¯\_(ツ)_/¯
    """

    def __init__(self):
        super().__init__()
        self.perceiver = Perceiver(
            input_channels = 512,          # number of channels for each token of the input
            input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = 3,                   # depth of net. The shape of the final attention mechanism will be:
                                         #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents = 16,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 512,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,         # number of dimensions per cross attention head
            latent_dim_head = 64,        # number of dimensions per latent self attention head
            num_classes = 512,          # output number of classes
            attn_dropout = 0.,
            ff_dropout = 0.,
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2      # number of self attention blocks per cross attention
        )

    def forward(self, x):
        res = self.perceiver(x.permute(0, 2, 3, 1))
        return res



class EncoderConv(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.activation = nn.ReLU()
        layers = [nn.Conv2d(n_in, 96, 3, 1, 1),
                  self.activation,
                  nn.MaxPool2d(3, 2),
                  nn.Conv2d(96, 192, 3, 1, 1),
                  self.activation,
                  nn.MaxPool2d(3, 2),
                  nn.Conv2d(192, 384, 3, 1),
                  self.activation,
                  nn.Conv2d(384, 384, 3, 1),
                  self.activation,
                  nn.Conv2d(384, n_out, 3, 1),
                  self.activation,
                  nn.MaxPool2d(3, 2)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        import pdb;pdb.set_trace()
        out = self.conv_block(x)
        return out


class AggFlat(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.activation = nn.ReLU()
        mid = 1000
        layers = [nn.Linear(n_in, mid),
                  nn.BatchNorm1d(mid),
                  self.activation,
                  nn.Linear(mid, mid),
                  nn.BatchNorm1d(mid),
                  self.activation,
                  nn.Linear(mid, n_out)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x.flatten(1))
        return out
