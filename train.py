import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn

from data import *
from model import *

# Parameters
epochs = 2000
batch_size = 64
learning_rate = 0.01

# Preparing Dataset
train = myDataset('./Dataset32/train', ToTensor)
test = myDataset('./Dataset32/test', ToTensor)
train_loader = DataLoader(train, batch_size=batch_size)
test_loader = DataLoader(test, batch_size=batch_size)

# Model initial
myModel = Model()
if torch.cuda.is_available():
    myModel = myModel.cuda()

# Loss function initial
loss_fun = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fun = loss_fun.cuda()

# Optimizer initial
optim = torch.optim.SGD(myModel.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    total_loss_in_train_set = 0
    for imgs, targets in train_loader:
        # cuda
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = myModel(imgs)
        loss = loss_fun(outputs, targets)
        total_loss_in_train_set += loss.item()
        # Optimize
        optim.zero_grad()
        loss.backward()
        optim.step()
    print('epoch = {}\t total_loss_train = {}'.format(epoch, total_loss_in_train_set))
    total_loss_in_test_set = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = myModel(imgs)
            loss = loss_fun(outputs , targets)
            total_loss_in_test_set += loss
    print('test_loss = {}'.format(total_loss_in_test_set))


