import torch
from torch import optim
import util
from infomax import *
from perceiver_pytorch import Perceiver
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
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


def tif_dataset():
    dataset_path = "/mnt/fileserver/shared/references/Biology/Genetic Data"
    tifs_path = glob.glob(dataset_path + '/*.tif')
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

    transform = lambda img: permute(resize(crop(img))) / 255.

    dataset = LargeTifDataset(n_items, tifs_path, transform)
    return dataset


def tiny_imagenet():
    dataset_path = "../tiny-imagenet-200/train/"
    dataset = TinyImageNet(dataset_path)
    return dataset


def main():
    epochs = 10
    n_items = 30
    snapshot_path = 'infomax.pt'
    batch_size = 55

    resnet34 = models.resnet34(pretrained=False)
    modules=list(resnet34.children())[:-2]

    # extracts local features
    resnet341 = torch.nn.Sequential(*modules)
    aggregator = AggregatorPerceiver()
    feature_map_size = 16 * 16
    feature_map_size = 2 * 2
    size_global_inp = 512 + 512 * feature_map_size
    global_loss = GlobalDiscriminatorFull(size_global_inp)
    local_loss = LocalDiscriminator(512, 512)

    opt_encoder = optim.AdamW([{'params': resnet341.parameters()},
                               {'params': aggregator.parameters()}], lr=0.00001)

    opt_global_discriminator = optim.AdamW(global_loss.parameters(), lr=0.00001)
    opt_local_discriminator = optim.AdamW(local_loss.parameters(), lr=0.00001)

    infomax = InfoMax(resnet341,
                      aggregator,
                      global_loss,
                      local_loss)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    infomax.to(device)
    global_loss.to(device)
    local_loss.to(device)
    infomax.train()

    dataset = tiny_imagenet()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    param_encoder = list(resnet341.parameters()) + list(aggregator.parameters())
    average = dict()
    t = 0
    for epoch in range(epochs):
        print('STARTING EPOCH ', epoch + 1)
        for i, batch in enumerate(loader):
            loss = infomax(batch.to(device))
            t += 1
            t = min(t, 100)

            # optimize resnet + encoder
            infomax.zero_grad()
            opt_encoder.zero_grad()
            # optimize model with respect to local discriminator
            loss['local_encoder_loss'].backward(inputs=param_encoder, retain_graph=True)
            loss['grad_resnet_l'] = resnet341[0].weight.grad.abs().max()

            # optimize model with respect to global discriminator
            loss['global_encoder_loss'].backward(inputs=param_encoder, retain_graph=True)
            loss['grad_resnet_g'] = resnet341[0].weight.grad.abs().max()
            opt_encoder.step()


            # optimize global discriminator
            infomax.zero_grad()
            opt_global_discriminator.zero_grad()
            loss['global_discriminator_loss'].backward(inputs=list(global_loss.parameters()),
                    retain_graph=True)
            loss['grad_global_disc'] = global_loss.layer0.weight.grad.abs().max()
            opt_global_discriminator.step()

            # optimize local discriminator
            infomax.zero_grad()
            opt_local_discriminator.zero_grad()
            loss['local_discriminator_loss'].backward(inputs=list(local_loss.parameters()))
            loss['grad_local_disc'] = local_loss.layer0.weight.grad.abs().max()
            opt_local_discriminator.step()

            util.update_average(average, t, {k: v.detach().item() for (k, v) in loss.items()})

            if i % 20 == 0:
                print()
                print('batch ', i)
                print(average)
                print()


            if i and i % 200 == 0:
                state_dict = {'resnet': resnet341.state_dict(),
                                  'aggregator': aggregator.state_dict(),
                                  'discriminator_local': local_loss.state_dict(),
                                  'discriminator_global': global_loss.state_dict()}
                # save
                torch.save(state_dict, snapshot_path)

#        dataset.reset()

if __name__ == '__main__':
   main()
