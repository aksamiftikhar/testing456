
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
import torch.nn as nn
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
import torchvision.transforms as transforms 


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



#%% Create the network for training
import torch.nn as nn
import torch.nn.functional as F
# Define the class Net
class Net(nn.Module):
    def __init__(self):    
    	# Define all the parameters of the net
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):   
    	# Do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#%% prepare the train/test loader for MNIST
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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=False, num_workers=0) 



#%% start the network training
import torch.optim as optim

# Instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()   
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
  
for batch_idx, data_target in enumerate(train_loader):
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # Complete a forward pass
    output = model(data)

    # Compute the loss, gradients and change the weights
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(batch_idx, ': ', data.shape)
    
#%% test the network performance
total, correct = 0, 0 
# Set the model in eval mode
model.eval()

for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    
    # Put each image into a vector
    inputs = inputs.view(-1, 28 * 28 * 1)
    
    # Do the forward pass and get the predictions
    outputs = model(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))

#%%

def MaxPool(input_image, kernel_size):
    if type(kernel_size)==tuple:
        w = kernel_size[0]
        h = kernel_size[1]
        return input_image[::w, ::h]
    else:
        
        input_image[0::kernel_size, ::kernel_size]


#%%
import numpy as np
arr = np.array([[1, 2, 3, 4], 
                [4, 5, 6, 7]
                ])

np.arr
# out = MaxPool(arr, 2)

# print(out)


# samples = random.sample(range(10),4)
# print(samples)



