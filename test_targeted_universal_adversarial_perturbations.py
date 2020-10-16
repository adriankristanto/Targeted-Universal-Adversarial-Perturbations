# reference: https://github.com/LTS4/DeepFool
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from targeted_universal_adversarial_perturbations import targeted_universal_adversarial_perturbations
import os
import sys

TOTAL_IMAGES = 100
# change the data path to the directory with all the images from the same class
DATA_PATH = '***'
CLEAN_CLASS = 88 # 88 for macaw
TARGET_CLASS = 130
MAX_ITER = 10

net = models.resnet34(pretrained=True)

# Switch to evaluation mode
net.eval()

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

# Remove the mean
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])

counter = 0
images = []
for image in os.listdir(DATA_PATH):
    if counter >= TOTAL_IMAGES:
        break
    # open the image and apply transform
    image = transform(Image.open(DATA_PATH + image))[None, :, :, :]

    # if it's predicted correctly, then add it to the images array
    if torch.argmax(net(image), dim=1).item() == CLEAN_CLASS:
        images.append(image)
        counter += 1

# create a single batch of size 100
images = torch.cat(images, dim=0)

universal_v = targeted_universal_adversarial_perturbations(images, net, TARGET_CLASS, MAX_ITER)

torch.save(universal_v, 'universal_perturbation.pth')
torchvision.utils.save_image(universal_v, 'universal_perturbation.png')

# test the perturbation
# im1 = Image.open('***')
# im1 = transform(im1)[None, :, :, :]

# universal_v = torch.load('universal_perturbation.pth')

# print(torch.argmax(net(im1)))
# print(f'prediction: {torch.argmax(net(im1 + universal_v))}')