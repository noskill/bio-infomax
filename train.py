import torch
from torch.utils.data import DataLoader
import cv2
import imutils
from transform import *
import re
import glob
import torchvision.models as models
from dataset import *
import tiffile


def main():
    modules=list(resnet34.children())[:-2]
    resnet34 = models.resnet34(pretrained=False)
    resnet341 = torch.nn.Sequential(*modules)



    n_items = 30 
    tifs_path = glob.glob('*.tif')
    crop = RandomCropTransform(size=1000, beta=250)
    resize = lambda img: imutils.resize(img, width=1000)
    transform = lambda img: resize(crop(img))
    dataset = LargeTifDataset(n_items, tifs_path, transform)
    loader = DataLoader(dataset, batch_size=3, shuffle=False) 
    for epoch in epochs:
        for batch in loader:
            loss = infomax(batch) 
            loss.backward()
            for opt in optimizers:
                opt.step()
