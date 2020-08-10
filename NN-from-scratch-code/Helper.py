import numpy as np #1.18.5
import json #2.0.9
import torch #1.6.0+cu101

class ActivationFunctionError(Exception):
    def __init__(self, message="Activation function can be either 'relu' or 'sigmoid'"):
        self.message = message
        super().__init__(self.message)

class ActivationDimensionError(Exception):
    def __init__(self, layerNum, message="Incorrect activation dimensions"):
        self.layerNum = layerNum
        self.message = message

    def __str__(self):
        return f'{self.message} at Layer {self.layerNum-1} -> Layer {self.layerNum}'

class ActivationError(Exception):
    def __init__(self, layerNum, message="Activations contain NaN/Inf values"):
        self.layerNum = layerNum
        self.message = message

    def __str__(self):
        return f'{self.message} at Layer {self.layerNum}'

class NNLayerCountError(Exception):
    def __init__(self, message="Network must have atleast 2 layers'"):
        self.message = message
        super().__init__(self.message)

class LabelDimensionError(Exception):
    def __init__(self, message="Labels should have dimensions (nClasses, nExamples)"):
        self.message = message
        super().__init__(self.message)

class Layer():
    def __init__(self, n_units, activation="relu", cuda=False):
        ''' 
        Class Variables:
        ----------------
        n_units             --  no. of neurons in the layer
        layerNum            --  layer number (L) of this layer in the network, starting from 0
        activationFunction  --  activation function
        weights             --  2D tensor storing the weights of the incoming edges to this layer
        biases              --  1D tensor storing the biases of the incoming edges to this layer
        activations         --  activations = g(Z), where g is the activation function given as input
        weighted_sum        --  weighted_sum = WA + b, where W and b are weight and biases tensors respectively 
                                A is the activation of previous layer
        cuda                --  boolean value indicating whether to use GPU

        Dimensions:
        -----------
        weights       --  (nL, nL-1)
        biases        --  (nL, 1)
        activations   --  (nL, m)
        weighted sum  --  (nL, m)
        '''
        self.n_units = n_units
        self.layerNum = None

        if activation not in ('relu', 'sigmoid'):
            raise ActivationFunctionError()

        self.activationFunction = activation
        self.weights = None
        self.biases = None
        self.activations = None
        self.weighted_sum = None
        self.cuda = cuda

    
    def forward(self, activations):
        '''
        Description:
        ------------
        Given, the activations from the previous layer, this function computes the weighted_sum 
        and applies the activation function on the calculated weighted_sum i.e

        weighted_sum = self.weights * activations(from prev layer) + biases
        activations = activationFunction(weighted_sum)

        Arguments:
        ----------
        activations -- (nL-1, m) tensor containing activations from the previous layer L-1
        '''

        ''' self.weights => (nL, nL-1), activations => (nL-1, m) '''
        if self.weights.shape[1]!=activations.shape[0]: 
            raise ActivationDimensionError(self.layerNum)

        ''' Move the activations to the gpu for computation '''
        if self.cuda:
            if not activations.is_cuda:
                activations = activations.cuda()

        ''' Cache weighted_sum, since during backpropagation it will be required to calculate d(activations)/d(weighted_sum)'''
        self.weighted_sum = torch.matmul(self.weights, activations)+self.biases

        ''' Calculate activations by applying the activation function on the weighted_sum '''
        if self.activationFunction == "sigmoid":
            self.activations = torch.sigmoid(self.weighted_sum)

        elif self.activationFunction == "relu":
            self.activations = torch.nn.functional.relu(self.weighted_sum, inplace=False)
          
        if torch.isnan(self.activations).any() or torch.isinf(self.activations).any():
            raise ActivationError(self.layerNum)

        return self.activations

    
    def backward(self):
        '''  
        Description:
        ------------
        In the forward pass, we calculated weighted_sum, and then applied the activation function on it i.e

        activations = activationFunction(weighted_sum)

        Now, we have to find d(activations)/d(weighted_sum).
        E.g: If y = 4x^2, then gradient = dy/dx = 8x
        '''

        if self.activationFunction=="sigmoid":
            gradients = torch.mul(self.activations, (1-self.activations))

        elif self.activationFunction=="relu":
            ones = torch.ones(self.weighted_sum.shape)
            zeros = torch.zeros(self.weighted_sum.shape)
            if self.cuda:
                ones = ones.cuda()
                zeros = zeros.cuda()
            gradients = torch.where(self.weighted_sum<0, zeros,ones)

        return gradients


class NeuralNet():

    def __init__(self, learningRate=0.01, cuda=True):
        '''  
        Class Variables:
        ----------------
        layerDims     --  python list which sequentially stores number of neurons in each layer of the network
        layers        --  python list which sequentially stores the Layer class objects
        nLayers       --  stores total number of layers in the network including the input and the output layer
        learningRate  --  stores the learning rate that is used to update parameters in backpropagation
        cuda          --  boolean value indicating whether to use GPU
        '''
        self.layerDims = []
        self.layers = []
        self.nLayers = 0
        self.learningRate = learningRate
        self.cuda = cuda


    def addLayer(self, layer):
        ''' 
        Description:
        ------------
        This function adds the Layer class object to the network.
        '''
        layer.cuda = self.cuda
        layer.layerNum = self.nLayers
        self.layerDims.append(layer.n_units)
        self.layers.append(layer)
        self.nLayers+=1

    
    def initialize(self, weights=None):
        '''  
        Description:
        ------------
        This function is used to initialize the weights and biases for all the layers of the network.
        
        Arguments:
        ----------
        weights -- python list of key value pairs which stores the weights and biases for every layer

        Note:
        -----
        If 'weights' argument is not passed, then the weights of each layer are initialised randomly.
        '''

        if len(self.layerDims)<=1:
            raise NNLayerCountError()
        
        ''' Layer 0 which is the input layer doesn't have weights and biases. '''
        if weights==None:
            for i in range(1, self.nLayers):
                layer = self.layers[i]
                
                ''' layer.weights => (nL, nL-1), layer.biases => (nL, 1) '''
                if self.cuda:
                    layer.weights = (torch.rand(self.layerDims[i], self.layerDims[i-1]) * 0.01).cuda()
                    layer.biases = torch.zeros(self.layerDims[i], 1).cuda()
                else:
                    layer.weights = torch.rand(self.layerDims[i], self.layerDims[i-1]) * 0.01
                    layer.biases = torch.zeros(self.layerDims[i], 1)
        else:
            for i in range(1, self.nLayers):
                layer = self.layers[i]
                layer.weights = weights[i]['weights']
                layer.biases = weights[i]['biases']


    def forward(self, trainingExamples):
        '''  
        Description:
        ------------
        This function executes the forward pass on the dataset and eventually generates activations at the output layer.

        Arguments:
        -----------
        trainingExamples -- (n, m) torch tensor containing m training examples, with each example having n features
        '''

        ''' Move the trainingExamples to the gpu for computation '''
        if self.cuda:
            if not trainingExamples.is_cuda:
                trainingExamples = trainingExamples.cuda()

        activation = trainingExamples
        for i in range(1, self.nLayers):
            layer = self.layers[i]
            activation = layer.forward(activation)

    
    def loss(self, labels):
        '''  
        Description:
        ------------
        This function returns the average loss (cross entropy loss) between activations from the output layer and the actual labels.
        '''
        outputLayerActivations = self.layers[-1].activations

        if labels.shape != outputLayerActivations.shape:
            raise LabelDimensionError()

        m = labels.shape[1]
        cost = -(1/m)*torch.sum(torch.mul(labels, torch.log(outputLayerActivations)) + torch.mul((1-labels),torch.log(1-outputLayerActivations)))  
        return cost


    def backward(self, labels):
        '''  
        This function implements backpropagation.
        It goes on calculating derivaties backwards and then updates the weights and biases accordingly.
        This is done so that the error/loss after the next forward pass is lower than the current error/loss.

        Arguments:
        ----------
        labels -- (nClasses, m) torch tensor containing actual labels for the dataset, 
                  where nClasses = No. of neurons in the output layer
        '''
        outputLayerActivations = self.layers[-1].activations

        ''' 
        Derivative for cross entropy loss function 
        L -- Loss 
        A -- Activations
        Z -- Weighted Sum
        W -- Weights
        B -- Biases
        '''

        ''' Calculating dL/dA '''
        dL_by_dA = -(torch.div(labels, outputLayerActivations) - torch.div(1 - labels, 1 - outputLayerActivations))
        
        m = self.layers[-1].activations.shape[1]

        for i in range(self.nLayers-1,0,-1):
            layer = self.layers[i]

            ''' Calculating dL/dZ '''
            dA_by_dZ = layer.backward()
            
            dL_by_dZ = torch.mul(dL_by_dA, dA_by_dZ)
            dZ_by_dA_prev = layer.weights.t()

            ''' Calculating dL/dA_prev '''
            dL_by_dA = torch.matmul(dZ_by_dA_prev, dL_by_dZ)
            
            ''' Calculating dL/dW '''
            dZ_by_dW = self.layers[i-1].activations.t()
            dL_by_dW = (1/m)*torch.matmul(dL_by_dZ, dZ_by_dW)

            ''' Calculating dL/dB '''

            ''' Note: Here dZ_by_dB = 1 '''
            dL_by_dB = 1 / m * (torch.sum(dL_by_dZ, dim = 1,keepdim = True)) #keepdims maintains the dimension of the resultant matrix
            
            layer.weights -=  self.learningRate*dL_by_dW
            layer.biases -= self.learningRate*dL_by_dB


    def train(self, dataset, labels, weights=None, printLossAfterEvery=50, nIterations=1000):
        ''' 
        Description:
        ------------ 
        This function is used to begin the training.
        Each pass calls forward(), loss() and backward() functions in sequence and then updates the weights and biases associated with each layer.

        Arguments:
        ----------
        dataset               --  (n, m) torch tensor containing m training examples with each example having n features
        labels                --  (nClasses, m) torch tensor containing labels for the m training examples
        weights               --  python list of key value pairs which stores the weights and biases for every layer.
                                  Used as a checkpoint to initialise weights to those where the training was stopped earlier
        printLossAfterEvery   --  number of iterations after which the loss and other details about training needs to be printed
        nIterations           --  number of iterations of the dataset for which training should continue and update the nework's parameters
        '''
        if self.cuda==True:
            dataset = dataset.cuda()
            labels = labels.cuda()

        ''' Initialize weights and biases all the layers in the network '''
        if weights==None:
            self.initialize()
        else:
            self.initialize(weights=weights)
        
        ''' Normalizing the dataset ''' 
        if (dataset>1).any():
            dataset = dataset/torch.max(dataset)
        
        ''' Activations at Layer-0/Input-Layer = Dataset (n, m) '''
        self.layers[0].activations = dataset
        
        ''' Forward and Backward pass '''
        for i in range(nIterations):
            self.forward(dataset)
            avgLoss = self.loss(labels).item()
            if i%printLossAfterEvery==0:
                print("Avg Loss after iteration "+str(i)+": "+str(avgLoss))
            self.backward(labels)
        
        ''' Bundle weights and biases into a package for return'''
        weights = [None,] #First layer is the input layer. Hence it doesn't contain any weights.
        for i in range(1, self.nLayers):
            layer = self.layers[i]
            weights.append({'weights':layer.weights, 'biases':layer.biases})

        return weights

    
    def saveModel(self, modelFilePath):
        '''  
        Description:
        ------------
        This function saves the model as a JSON file.
        '''
        model = dict()

        ''' Save parameters of the input layer of the network '''
        layer = self.layers[0]
        model["0"] = {"num-units":layer.n_units, "activation-function": layer.activationFunction}
        model["learning-rate"] = self.learningRate
        model["num-layers"] = self.nLayers

        ''' Save parameters for the rest of the layers of the network '''
        
        ''' If the model is trained on gpu, shift the tensors to cpu first. '''
        if self.cuda:
            for i in range(1, self.nLayers):
                layer = self.layers[i]
                model[str(i)] = {"num-units":layer.n_units, "activation-function": layer.activationFunction, \
                                "weights": layer.weights.cpu().tolist(), "biases": layer.biases.cpu().tolist()}
        else:
            for i in range(1, self.nLayers):
                layer = self.layers[i]
                model[str(i)] = {"num-units":layer.n_units, "activation-function": layer.activationFunction, \
                                "weights": layer.weights.tolist(), "biases": layer.biases.tolist()}

        with open(modelFilePath, 'w') as fp:
            json.dump(model, fp, indent=4)

    
    def loadModel(self, modelFilePath):
        '''  
        Description:
        ------------
        This function is used to load the model saved in JSON format.
        '''
        with open(modelFilePath, "r") as f:
            model = json.load(f) 

        ''' Initialise new model and get the parameters from the saved model'''
        if len(self.layers)!=0: 
            self.__init__()
        self.learningRate = model["learning-rate"]
        
        ''' Add the input layer to the newly created model and update its parameters from the saved model '''
        n_units = model["0"]["num-units"]
        activationFunction = model["0"]["activation-function"]
        layerObj = Layer(n_units=n_units, activation=activationFunction)
        layerObj.cuda = self.cuda
        self.addLayer(layerObj)
        
        ''' Add other layers to the newly created model and update their parameters from the saved model '''
        for i in range(1, model["num-layers"]):
            n_units = model[str(i)]["num-units"]
            activationFunction = model[str(i)]["activation-function"]
            weights = model[str(i)]["weights"]
            biases = model[str(i)]["biases"]
            layerObj = Layer(n_units=n_units)
            layerObj.activationFunction = activationFunction
            
            if self.cuda:
                layerObj.weights = torch.FloatTensor(weights).cuda()
                layerObj.biases = torch.FloatTensor(biases).cuda()
            else:
                layerObj.weights = torch.FloatTensor(weights)
                layerObj.biases = torch.FloatTensor(biases)

            layerObj.cuda = self.cuda
            self.addLayer(layerObj)

        ''' Bundle weights and biases into a package for return'''
        weights = [None,] #First layer is the input layer. Hence it doesn't contain any weights.
        for i in range(1, self.nLayers):
            layer = self.layers[i]
            weights.append({'weights':layer.weights, 'biases':layer.biases})

        return weights


    def predict(self, example, singleExample=True):
        '''  
        Description:
        ------------
        This function is used to predict the label for the example given as input.
        It just makes use of the forward pass.

        Arguments:
        ----------
        example         --  (n, m)/(n,1) torch tensor containing m/1 examples, each having n features
        singleExample   --  boolean value, indicating whether a single example or a batch of example needs to be predicted
        '''
        shape = example.shape
        if len(shape)==1:
            raise ActivationDimensionError(0)

        try:
            _ = example.shape[1]
        except:
            example = example.reshape(example.shape[0],1)

        ''' Normalize the example and feed it to the network'''
        example = example/torch.max(example)
        self.forward(example)

        ''' Get output layer activations '''
        activations = self.layers[-1].activations

        if singleExample:
            ''' Reshape tensor of dimension (n,1) to n '''
            predictions = activations.reshape(activations.shape[0])
            prediction = torch.argmax(predictions).item()
            confidence = torch.max(predictions).item()

        else:
            confidence, prediction = torch.max(activations, dim=0)

        return prediction, confidence 