import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision #torchvision==0.7.0
import torchvision.transforms as transforms

PATH = "/content/model.pth"
PATH_TO_STORE_DATASET = "/content/data"

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


    def loadData(self):
        '''  
        Description:
        ------------
        This function downloads the MNIST dataset and prepares it for training using the torch DataLoaders
        '''
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])])

        trainset = torchvision.datasets.MNIST(root=PATH_TO_STORE_DATASET, train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

        testset = torchvision.datasets.MNIST(root=PATH_TO_STORE_DATASET, train=False, download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)


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

    def train(self, epochs=10):
        '''  
        Description:
        ------------
        This function is used to train the data.
        '''
        
        lossFunction = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            """ Loss at beginning of each iteration is set to 0 """
            combinedLoss = 0.0

            ''' i represents the batch number '''
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data

                if self.gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                ''' Clear the previously calculated gradients '''
                optimizer.zero_grad()

                ''' Forward pass '''
                outputs = self(inputs)

                ''' Calculating loss at the output layer between output layer activations and the actual labels '''
                batchLoss = lossFunction(outputs, labels)

                ''' Backward pass '''
                batchLoss.backward()

                ''' Updating the parameters '''
                optimizer.step()

                combinedLoss += batchLoss.item()

            print('Epoch:%d, Loss: %.3f' %(epoch + 1, combinedLoss/len(self.trainloader)))


    def test(self):
        '''  
        Description:
        ------------
        This function is used to evaluate the performance of the network on the test dataset.
        '''
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                if self.gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network: %d %%' % (100 * correct / total))


n = ConvNet(lr=0.001, gpu=True).cuda()
n.loadData()
n.lr = 0.0005
n.train(epochs=10)

n.test()

state_dict = n.state_dict()
torch.save(state_dict, PATH)