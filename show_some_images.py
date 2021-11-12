#from scipy.misc import face
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# step 1: input data(x and y)
image = np.loadtxt("image-train.txt")
label = np.loadtxt("label-train.txt")
image = torch.from_numpy(image).float()
image = image.view(-1, 3, 32, 32)
label = torch.from_numpy(label).float()
# after reshaping, x should have N rows where N is the number of samples
# y should also have N rows(entries) since each sample point has one label

# capture and draw the first twenty images and print their labels
img_draw = image[0:20,:]
#img_draw = img_draw * 100
l_draw = label[0:20]
print("Can we draw?", img_draw[0:3, 0:3, 0:4, 0:4])

def show(img):
    npimg = img.numpy().astype(int)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

grid = torchvision.utils.make_grid(img_draw, nrow=5, padding=10)
show(grid)

# grid_img = torchvision.utils.make_grid(img_draw, nrow=5)
# plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
print("The first 20 labels are:", l_draw)
