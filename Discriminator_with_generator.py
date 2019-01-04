#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:56:00 2018

@author: kavjit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
import h5py
from random import randint
import time 
import torch.autograd as autograd
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#parameters
batch_size = 128
n_z = 100
n_classes = 10
num_epochs = 200
gen_train = 1   #how often generator should be trained



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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=32)

#################################DEFINING DISCR AND GEN########################################################
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
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(196,1)
        self.fc10 = nn.Linear(196,10)
        
    def forward(self,x):
        #print(x.size())
        x = F.leaky_relu(self.layernorm1(self.conv1(x)))
        x = F.leaky_relu(self.layernorm2(self.conv2(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.layernorm3(self.conv3(x)))
        x = F.leaky_relu(self.layernorm4(self.conv4(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.layernorm5(self.conv5(x)))
        x = F.leaky_relu(self.layernorm6(self.conv6(x)))
        x = self.dropout3(x)
        x = F.leaky_relu(self.layernorm7(self.conv7(x)))
        x = F.leaky_relu(self.layernorm8(self.conv8(x)))
        x = self.dropout4(x)
        x = self.pool(x)
        x = x.view(-1, 196)
        x1 = self.fc1(x)    #Relu after this?
        x10 = self.fc10(x)
        return [x1,x10]

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(100,3136)
        self.batchnorm1 = nn.BatchNorm1d(3136)
        self.conv1 = nn.ConvTranspose2d(196,196,4, padding =1, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(196)
        self.conv2 = nn.Conv2d(196,196,3, padding =1, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(196)
        self.conv3 = nn.Conv2d(196,196,3, padding =1, stride=1)
        self.batchnorm4 = nn.BatchNorm2d(196)
        self.conv4 = nn.Conv2d(196,196,3, padding =1, stride=1)
        self.batchnorm5 = nn.BatchNorm2d(196)
        self.conv5 = nn.ConvTranspose2d(196,196,4, padding =1, stride=2)
        self.batchnorm6 = nn.BatchNorm2d(196)
        self.conv6 = nn.Conv2d(196,196,3, padding =1, stride=1)
        self.batchnorm7 = nn.BatchNorm2d(196)
        self.conv7 = nn.ConvTranspose2d(196,196,4, padding =1, stride=2)
        self.batchnorm8 = nn.BatchNorm2d(196)
        self.conv8 = nn.Conv2d(196,3,3, padding =1, stride=1)
        
        
        
    def forward(self,x):
        x = self.fc1(x)    #batchnorm after FC???
        x = self.batchnorm1(x)
        x = x.view(-1,196,4,4)
        x = F.relu(self.batchnorm2(self.conv1(x)))
        x = F.relu(self.batchnorm3(self.conv2(x)))
        x = F.relu(self.batchnorm4(self.conv3(x)))
        x = F.relu(self.batchnorm5(self.conv4(x)))
        x = F.relu(self.batchnorm6(self.conv5(x)))
        x = F.relu(self.batchnorm7(self.conv6(x)))
        x = F.relu(self.batchnorm8(self.conv7(x)))
        x = torch.tanh(self.conv8(x))
        return x
###########################################################################################################
###########################################################################################################

########################################  DEFINING FUNCTIONS  #################################################        

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig
###########################################################################################################
###########################################################################################################


aD =  Discriminator()
aG = Generator()
#aD.load_state_dict(torch.load('D_chkpt.ckpt'))
#aG.load_state_dict(torch.load('G_chkpt.ckpt'))
aG.cuda()
aD.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

criterion = nn.CrossEntropyLoss()

np.random.seed(352)
label = np.asarray(list(range(10))*10) #generates a 100x1 array repeating 0-9
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)] #there are 10 examples for each class
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()





###########################################################################################################
###########################################################################################################
# Train the model
for epoch in range(0,num_epochs):
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []
    start_time = time.time()
  
    aG.train()
    aD.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0] < batch_size):
            continue
        
        ###### train G
        #aD.eval()
        #aG.train()
        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(False)
        
            aG.zero_grad()
        
            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()
        
            fake_data = aG(noise)
            gen_source, gen_class  = aD(fake_data)
        
            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)
        
            gen_cost = -gen_source + gen_class
            gen_cost.backward()
            for group in optimizer_g.param_groups:
                for p in group['params']:
                    state = optimizer_g.state[p]
                    if('step' in state and state['step']>=1024):
                        state['step'] = 1000
            optimizer_g.step()
           
            
        ##### train D
        #aD.train()
        #aG.eval()
        for p in aD.parameters():
            p.requires_grad_(True)
        
        aD.zero_grad()
        
        # train discriminator with input from generator
        label = np.random.randint(0,n_classes,batch_size)
        noise = np.random.normal(0,1,(batch_size,n_z))
        label_onehot = np.zeros((batch_size,n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)
        
        disc_fake_source, disc_fake_class = aD(fake_data)
        
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)
        
        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()
        
        disc_real_source, disc_real_class = aD(real_data)
        
        prediction = disc_real_class.data.max(1)[1]
        accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0
        
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)
        
        gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)
        
        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()
        for group in optimizer_d.param_groups:
            for p in group['params']:
                state = optimizer_d.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000        
        optimizer_d.step()
        
        # within the training loop
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        if((batch_idx%50)==0):
            print(epoch, batch_idx, "%.2f" % np.mean(loss1), 
                                    "%.2f" % np.mean(loss2), 
                                    "%.2f" % np.mean(loss3), 
                                    "%.2f" % np.mean(loss4), 
                                    "%.2f" % np.mean(loss5), 
                                    "%.2f" % np.mean(acc1))
    
    # Test the model
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()
    
            with torch.no_grad():
                _, output = aD(X_test_batch)
    
            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing accuracy of D at epoch {} : {} time: {}'.format(epoch,accuracy_test,time.time()-start_time))           
    
    ### save output
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()
    
    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)
    
    if(((epoch+1)%1)==0):
        torch.save(aG,'generator.model')
        torch.save(aD,'discriminator.model')
        torch.save(aG.state_dict(),'G_chkpt.ckpt')
        torch.save(aD.state_dict(),'D_chkpt.ckpt')

###########################################################################################################
###########################################################################################################
print('finished training')
torch.save(aG,'generator.model')
torch.save(aD,'discriminator.model')
torch.save(aG.state_dict(),'G_chkpt.ckpt')
torch.save(aD.state_dict(),'D_chkpt.ckpt')


# Test the model
aD.eval()
with torch.no_grad():
    test_accu = []
    for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
        X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()

        with torch.no_grad():
            _, output = aD(X_test_batch)

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
        test_accu.append(accuracy)
        accuracy_test = np.mean(test_accu)
print('Testing ',accuracy_test)



