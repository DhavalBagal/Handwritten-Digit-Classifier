import numpy as np #1.18.5
#import json #2.0.9
import torch #1.6.0+cu101
from Helper import NeuralNet


def printDigitOnTerminal(img):
    if type(img)==list:
        temp = np.array(img)
    elif type(img)==np.ndarray:
        temp = img    
    xmax, xmin = temp.max(), temp.min()
    temp = (temp - xmin)/(xmax - xmin)
    temp = temp.tolist()

    for x in temp:
        for y in x:
            if y==0.0:
                print(" ", end=" ")
            else:
                print("1", end=" ")
        print()


class DigitRecognizer():
    def __init__(self, path, cuda=False):
        '''  
        Description:
        ------------
        This function loads the model from the json file
        '''
        self.nn = NeuralNet(cuda=False)
        self.nn.loadModel(path)

    def predict(self, imgVec):
        '''  
        Arguments:
        ----------
        imgVec -- 2d list representing the single channel image
        '''
        printDigitOnTerminal(imgVec)
        imgVec = torch.FloatTensor(imgVec)
        ''' Normalizing the image so that all values are between 0 and 1 '''
        imgVec = imgVec/torch.max(imgVec)
        
        imgVec = imgVec.reshape(imgVec.shape[0]*imgVec.shape[0],1)
        
        return self.nn.predict(imgVec, singleExample=True)