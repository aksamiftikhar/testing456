# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:36:19 2022

@author: Abdullah
"""
import torch
import numpy as np

t = torch.tensor([[1, 2]])

t2 = torch.rand(2,2)

t3 = torch.rand((3,4))

def sq(t): 
    return list(map(lambda x: x*x, t))

print(t)
print(t2)
print(t3)

t4 = torch.rand(t2.shape)
t5 = torch.tensor([[1,2], 
                   [2, 3]])
print(t5.matmul(t5))

t6 = torch.zeros(3, 4)
print(t6)

narr = t4.numpy()
t7 = torch.from_numpy(narr)
t8 = torch.tensor(narr)
print(t5)


print(narr, t7, sq(t5))


#%%
import torch

# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q = x + y
q.retain_grad()
f = q * z

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))
print("Gradient of z is: " + str(q.grad))

#%%

import torch

inpt = torch.rand((784,))
w1 = torch.rand(784, 200)
w2 = torch.rand(200, 10)

h1 = torch.matmul(inpt, w1)
output = torch.matmul(h1, w2)

#%%


import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    
#%%
import torch.nn
# Initialize the scores and ground truth
logits = torch.tensor([[-1.2, 0.12, 4.8]])
ground_truth = torch.tensor([2])

# Instantiate cross entropy loss
criterion = nn.CrossEntropyLoss()

# Compute and print the loss
loss = criterion(logits, ground_truth)
print(loss) 

#%% prepare and loading datasets

import torch
import torchvision
import torch.utils.data
import torchvision.transformers as transformers 


# Transform the data to torch tensors and normalize it 
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.1307), ((0.3081)))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, 
									  download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False,
			   download=True, transform = transform)


# Prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=True, num_workers=0) 



#%% loading the MNIST trainging/test datasets

import torch
import torchvision.datasets
import torch.utils.data
from torchvision import transforms

# Transform the data to torch tensors and normalize it 
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.1307), ((0.3081)))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, 
									  download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False,
			   download=True, transform = transform)


# Prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=False, num_workers=0) 





# inspect the loaders
# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# Print the computed shapes
print(trainset_shape, testset_shape)

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size

# Print sizes of the minibatch
print(trainset_batchsize, testset_batchsize)