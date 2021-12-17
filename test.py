import torch
import cv2
import imutils
from transform import *
import re
import glob
import torchvision.models as models
from dataset import *
import tiffile

#mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)

mobilenet_v2 = models.mobilenet_v2(pretrained=True)
model = mobilenet_v2
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

resnet34 = models.resnet34(pretrained=False)





# 2000x2000 requires 3400mb with backpropagation on googlenet

modules=list(resnet34.children())[:-2]
resnet341 = torch.nn.Sequential(*modules)

crop = RandomCropTransform(size=1000, beta=250)

paths = glob.glob('*.tif') 
for path in paths:
    print('\n')
    print(path)
    result = load_tif(path)
    one_gb_image = find_closest(result, 1000)
    import pdb;pdb.set_trace()
    img = page2array(one_gb_image[1])
    img_crop = crop(img)
    resized = imutils.resize(img_crop, width=1000)
    cv2.imshow('crop', resized)
    cv2.waitKey()

