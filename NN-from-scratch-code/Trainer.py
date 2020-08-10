import numpy as np #1.18.5
import tensorflow as tf #2.3.0
import torch #1.6.0+cu101
from keras.utils import to_categorical #2.4.3
from Helper import NeuralNet

class MNISTTrainer():
    def __init__(self):
        ''' 
        Initial Dimensions:
          - x_train: (60000, 28, 28)
          - x_test: (10000, 28, 28)
          - y_train: (60000,)
          - y_test: (10000,)
        '''
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        ''' Normalizing data '''
        x_train = x_train/255.0   #(784, 60000)
        x_test = x_test/255.0     #(784, 10000)

        self.plain_test_labels = np.array(y_test)

        ''' Converting 2D Images into 1D vectors '''
        x_train, x_test = x_train.T.reshape((784,60000)), x_test.T.reshape((784,10000))

        ''' One hot encoding '''
        train_labels = to_categorical(y_train)  #(60000, 10)
        test_labels = to_categorical(y_test)    #(10000, 10)
        
        ''' Convert labels to numpy arrays '''
        y_train = np.array(train_labels).T  #(10, 60000)
        y_test = np.array(test_labels).T    #(10, 10000)

        ''' 
        Dimensions after processing:
          - x_train :  (784, 60000),  784 = 28px * 28px
          - x_test  :  (784, 10000),  784 = 28px * 28px
          - y_train :  (10, 60000),   10  = No. of classes 
          - y_test  :  (10, 10000),   10  = No. of classes
        '''
        
        ''' Converting to torch tensors '''
        self.x_train = torch.from_numpy(x_train).float()
        self.y_train = torch.from_numpy(y_train).float()
        self.x_test = torch.from_numpy(x_test).float()
        self.y_test = torch.from_numpy(y_test).float()


    def loadData(self):
        ''' Return the dataset in the format required by the rest of the code '''
        return self.x_train, self.y_train, self.x_test, self.y_test, self.plain_test_labels


    def setUpNetwork(self, savePath, cuda):
        self.n = NeuralNet(learningRate=0.05, cuda=cuda)
        self.n.loadModel(savePath)


    def train(self, savePath, weights=None):
        if weights==None:
            self.setUpNetwork(savePath, True)
        try:
            for i in range(5):
                weights =  self.n.train(self.x_train, self.y_train, nIterations=500, printLossAfterEvery=100, weights=weights)
                self.n.saveModel(savePath)
        except Exception as e:
            pass

        return weights

    
    def test(self, savePath, weights=None):
        if weights==None:
            self.setUpNetwork(savePath, False)
        labels = torch.from_numpy(self.plain_test_labels).int()

        self.x_test = self.x_test/255.0
        prediction, confidence = self.n.predict(self.x_test, singleExample=False)
        prediction = prediction.int()

        count = 0
        total = prediction.shape[0]

        print("\n\n", prediction,"\n\n", labels, "\n\n")
        for i in range(total):
            if prediction[i] == labels[i]:
                count+=1
        print("\nTest Set Accuracy: "+str((count/total)*100)+"%")

a = MNISTTrainer()
a.test("/Users/dhavalbagal/Desktop/ACTIVITIES/BAGAL/Digit-Classifier/model.json")
"""  
a = MNISTTrainer()
weights = a.train("/content/model.json", weights=None)
weights = a.train("/content/model.json", weights=weights)

a = MNISTTrainer()
a.test("/content/model.json")
"""