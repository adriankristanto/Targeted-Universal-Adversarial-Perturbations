# reference: https://github.com/LTS4/DeepFool
# tested using pytest
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
from targeted_deepfool import targeted_deepfool
import os

net = models.resnet34(pretrained=True)

# Switch to evaluation mode
net.eval()

im_orig = Image.open('test_im1.jpg')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]


# Remove the mean
im = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

def test_1():
    v_total, i, perturbed_image = targeted_deepfool(image=im, net=net, target_class=130)
    assert torch.argmax(net(perturbed_image)) == 130

def test_2():
    v_total, i, perturbed_image = targeted_deepfool(image=im, net=net, target_class=1)
    assert torch.argmax(net(perturbed_image)) == 1

def test_3():
    v_total, i, perturbed_image = targeted_deepfool(image=im, net=net, target_class=100)
    assert torch.argmax(net(perturbed_image)) == 100

def test_4():
    v_total, i, perturbed_image = targeted_deepfool(image=im, net=net, target_class=31)
    assert torch.argmax(net(perturbed_image)) == 31