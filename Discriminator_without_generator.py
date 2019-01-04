import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


import numpy as np
import h5py
from random import randint
import time 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3,196,3,padding = 1, stride = 1)
        self.layernorm1 = nn.LayerNorm([196,32,32])
        self.conv2 = nn.Conv2d(196,196,3,padding = 1, stride = 2)
        self.layernorm2 = nn.LayerNorm([196,16,16])
        self.conv3 = nn.Conv2d(196,196,3,padding = 1, stride = 1)
        self.layernorm3 = nn.LayerNorm([196,16,16])
        self.conv4 = nn.Conv2d(196,196,3,padding = 1, stride = 2)
        self.layernorm4 = nn.LayerNorm([196,8,8])
        self.conv5 = nn.Conv2d(196,196,3,padding = 1, stride = 1)
        self.layernorm5 = nn.LayerNorm([196,8,8])
        self.conv6 = nn.Conv2d(196,196,3,padding = 1, stride = 1)
        self.layernorm6 = nn.LayerNorm([196,8,8])
        self.conv7 = nn.Conv2d(196,196,3,padding = 1, stride = 1)
        self.layernorm7 = nn.LayerNorm([196,8,8])
        self.conv8 = nn.Conv2d(196,196,3,padding = 1, stride = 2)
        self.layernorm8 = nn.LayerNorm([196,4,4])
        
        self.pool = nn.MaxPool2d(kernel_size = 4,stride = 4)
        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)
        
    def forward(self,x):
        #print(x.size())
        x = F.leaky_relu(self.layernorm1(self.conv1(x)))
        x = F.leaky_relu(self.layernorm2(self.conv2(x)))
        x = F.leaky_relu(self.layernorm3(self.conv3(x)))
        x = F.leaky_relu(self.layernorm4(self.conv4(x)))
        x = F.leaky_relu(self.layernorm5(self.conv5(x)))
        x = F.leaky_relu(self.layernorm6(self.conv6(x)))
        x = F.leaky_relu(self.layernorm7(self.conv7(x)))
        x = F.leaky_relu(self.layernorm8(self.conv8(x)))
        #print(x.size())
        x = self.pool(x)
        #print(x.size())
        x = x.view(-1, 196)
        
        x1 = self.fc1(x)    #Relu after this?
        x10 = self.fc10(x)
        
        return [x1,x10]


model = Discriminator()
model.to(device)
#model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

#hyperparameters
epochs = 100
learning_rate = 0.0001

for epoch in range(epochs):
    correct = 0
    total = 0
    if epoch==50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10
    if epoch==75:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100            
                        
    for i,data in enumerate(trainloader):
        inputs,labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output1, output10 = model(inputs)
        loss = criterion(output10,labels)
        
        loss.backward()
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        
        optimizer.step()
        
        _, predicted = torch.max(output10.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #print('hellloo')
    
    print('accuracy at epoch {} = {}'.format(epoch,correct/total))
    if epoch%5 == 0:
        torch.save(model.state_dict(),'checkpoint.ckpt')
        torch.save(model,'full_model.ckpt')

    
torch.save(model.state_dict(),'D1_chkpt.ckpt')
torch.save(model,'discriminator1.model')    
    
print('Finished Training')



correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
    #for i,data in enumerate(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        output1, output10 = model(images)
        _, predicted = torch.max(output10.data, 1)   #output.data? #_for ignoring a value while unpacking second return is argmax index, 1 to indicate axis to check
        total += labels.size(0) #why?
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))



















