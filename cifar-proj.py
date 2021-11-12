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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128 * 2 * 2, 120)  # 2*2 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        # all dimensions except the batch/sample index dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



# input data(x and y)
image = np.loadtxt("image-train.txt")
label = np.loadtxt("label-train.txt")
image = torch.from_numpy(image).float()
image = image.view(-1, 3, 32, 32)
label = torch.from_numpy(label).float()
# after reshaping, x should have N rows where N is the number of samples
# y should also have N rows(entries) since each sample point has one label

# create net and optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# split the whole data into 400 batches, each batch has 100 data points
batch_size = 100
batches = image.size()[0] / batch_size
batches = int(batches)

for epoch in range(10):

    running_loss = 0.0
    for batch in range(batches):

        #extract certain batch of data
        head = batch * batch_size
        head = int(head)
        tail = (batch + 1) * batch_size
        tail = int(tail)
        image_little = image[head:tail, :]
        label_little = label[head:tail]

        optimizer.zero_grad()   # zero the gradient buffers
        output = net(image_little)
        target = label_little.numpy().astype(int)
        target_mtx = np.eye(10)[target]
        target_mtx = torch.from_numpy(target_mtx).float()
        loss = criterion(output, target_mtx)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch % 40 == 39:    # print every 40 mini-batches
            params = list(net.parameters())
            #print("after", head, "and", tail, "the first param is", params[0][0])
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch + 1, running_loss / 40))
            running_loss = 0.0

print("Finished Training")

#Save the train model
PATH = './proj_testnet.pth'
torch.save(net.state_dict(), PATH)
#Load trained model for test
tnet = Net()
tnet.load_state_dict(torch.load(PATH))

#Load test image
image_test = np.loadtxt("image-test.txt")
image_test = torch.from_numpy(image_test).float()
image_test = image_test.view(-1, 3, 32, 32)
batches_test = image_test.size()[0] / batch_size
batches_test = int(batches_test)
result = np.array([0.387])

for batch in range(batches_test):

    head = batch * batch_size
    head = int(head)
    tail = (batch + 1) * batch_size
    tail = int(tail)
    image_test_little = image_test[head:tail, :]

    toutput = tnet(image_test_little)
    row_sm = nn.Softmax(dim = 1)
    prediction = row_sm(toutput) #after softmax, a 100*10 matrix
    pred_label = torch.max(prediction, dim = 1).indices.numpy()
    result = np.append(result, pred_label)

    if batch % 20 == 19:
        print("result added to test batch number:", batch)

print("size of numpy output is:", result.shape)
# save results
a_file = open("y.out", "w")
np.savetxt(a_file, result)
a_file.close()
