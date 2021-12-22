import torch
from torch import optim
import util
from infomax import *
import torch.nn as nn
from agg import *
from torch.utils.data import DataLoader
import cv2
from transform import *
from tiny import *
import re
import glob
import torchvision.models as models
from dataset import *
import tiffile


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
    dataset_path = "../tiny-imagenet-200/"
    dataset = TinyImageNet(dataset_path, SETType.TRAIN)
    return dataset


def main():
    epochs = 10
    n_items = 30
    snapshot_path = 'infomax.pt'
    batch_size = 55

    resnet34 = models.resnet18(pretrained=False)
    modules=list(resnet34.children())[:-3]

    # extracts local features
    resnet341 = torch.nn.Sequential(*modules)
    aggregator = AggregatorPerceiver()
    aggregator = AggFlat(256 * 4 * 4, 512)
    feature_map_size = 16 * 16
    feature_map_size = 4 * 4
    size_global_inp = 512 + 256 * feature_map_size
    global_loss = GlobalDiscriminatorFull(size_global_inp)
    local_loss = LocalDiscriminatorConv(256, 512)

    opt_encoder = optim.RMSprop([{'params': resnet341.parameters()},
                               {'params': aggregator.parameters()}], lr=0.0001)

    opt_global_discriminator = optim.RMSprop(global_loss.parameters(), lr=0.00005)
    opt_local_discriminator = optim.RMSprop(local_loss.parameters(), lr=0.00005)

    tiny_class = TinyClass(512, 200)
    tiny_opt = optim.RMSprop(tiny_class.parameters(), lr=0.001)

    infomax = InfoMax(resnet341,
                      aggregator,
                      global_loss,
                      local_loss)


    device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(snapshot_path):
        print('loading from ', snapshot_path)
        state_dict = torch.load(snapshot_path, map_location=device)
        resnet341.load_state_dict(state_dict['resnet'])
        aggregator.load_state_dict(state_dict['aggregator'])
        local_loss.load_state_dict(state_dict['discriminator_local'])
        global_loss.load_state_dict(state_dict['discriminator_global'])
    if os.path.exists('tiny.pt'):
        tiny_class.load_state_dict(torch.load('tiny.pt'))
    tiny_class.to(device)
    infomax.to(device)
    global_loss.to(device)
    local_loss.to(device)
    infomax.train()

    dataset = tiny_imagenet()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    param_encoder = list(resnet341.parameters()) + list(aggregator.parameters())
    average = dict()
    t = 0
    for epoch in range(epochs):
        print('STARTING EPOCH ', epoch + 1)
        for i, batch in enumerate(loader):
            data, labels = batch
            loss, features = infomax(data.to(device))
            label_est = tiny_class(features)
            tiny_loss = - label_est[torch.arange(len(labels)), labels].mean()
            loss['tiny_loss'] = tiny_loss
            loss['tiny_acc'] = (torch.argmax(label_est, dim=1).to(labels) == labels).float().mean()

            if epoch == 0 and i == 0:
                print({k: v.item() for k, v in loss.items()})

            # optimize supervised classifier
            tiny_opt.zero_grad()
            tiny_loss.backward(inputs=list(tiny_class.parameters()))
            tiny_opt.step()

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

          #  # optimize global discriminator
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
                print({k: v.item() for k, v in loss.items()})
                state_dict = {'resnet': resnet341.state_dict(),
                               'aggregator': aggregator.state_dict(),
                               'discriminator_local': local_loss.state_dict(),
                               'discriminator_global': global_loss.state_dict()}
                # save
                torch.save(state_dict, snapshot_path)
                torch.save(tiny_class.state_dict(), 'tiny.pt')

#        dataset.reset()

if __name__ == '__main__':
   main()
