import torch
import util
from infomax import *
from perceiver_pytorch import Perceiver
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import imutils
from transform import *
import re
import glob
import torchvision.models as models
from dataset import *
import tiffile


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
            depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
                                         #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
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


def main():
    epochs = 30
    resnet34 = models.resnet34(pretrained=False)
    modules=list(resnet34.children())[:-2]

    # extracts local features
    resnet341 = torch.nn.Sequential(*modules)
    aggregator = AggregatorPerceiver()
    size_global_inp = 512 + 512 * 16 * 16
    global_loss = GlobalDiscriminatorFull(size_global_inp)
    local_loss = LocalDiscriminator(512, 512)

    opt_encoder = optim.Adam([{'params': resnet341.parameters()},
                              {'params': aggregator.parameters()}], lr=0.0001) 

    opt_discriminator = optim.Adam([{'params': global_loss.parameters()},
                                    {'params': local_loss.parameters()}], lr=0.0001)
    optimizers = dict(encoder_loss=opt_encoder,
                      discriminator_loss=opt_discriminator)                      

    infomax = InfoMax(resnet341,
                      aggregator,
                      global_loss,
                      local_loss)
    n_items = 30
    batch_size = 3
    tifs_path = glob.glob('*.tif')
    crop_size = 500
    crop = RandomCropTransform(size=crop_size, beta=crop_size // 4)
    def resize(img):
        res = util.resize(img, height=crop_size, width=crop_size)
        if res.shape != (crop_size, crop_size, 3):
            import pdb;pdb.set_trace()
            res = util.resize(img, height=crop_size, width=crop_size)
        return res

    def permute(img):
        return numpy.moveaxis(img, (0, 1, 2), (1, 2, 0))

    transform = lambda img: permute(resize(crop(img)))

    dataset = LargeTifDataset(n_items, tifs_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    for epoch in range(epochs):
        for batch in loader:
            loss = infomax(batch / 255.)
            for opt in optimizers.values():
                opt.zero_grad()
            for k, l in loss.items():
                # we need to backpropagate few times
                l.backward(retain_graph=True)
            for opt in optimizers:
                opt.step()
        dataset.reset()

if __name__ == '__main__':
   main()

