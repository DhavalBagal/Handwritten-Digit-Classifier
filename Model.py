import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms

#PATH_TO_STORE_DATASET = "/content/data"

class ConvNet(nn.Module):

    def __init__(self, lr, gpu=True):
        '''  
        Description:
        ----------
        This function builds the structure of the network i.e it declares all the layers which form the network
        '''
        super(ConvNet, self).__init__()
        self.lr = lr
        self.conv1 = nn.Conv2d(1, 6, 5) #kernel=(5,5), input_channels=1, output_channels=6
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.gpu = gpu

    def forward(self, x):
        '''  
        Description:
        ------------
        This function specifies the forward pass i.e the sequence of operations on the input data to product activations 
        at the output layer.
        '''
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        ''' Flatten the 3d activations into a 1d vector to feed it to the fully connected layers'''
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

