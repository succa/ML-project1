# -*- coding: utf-8 -*-
""" Implementation of the layers. """

#from torch import FloatTensor
import numpy as np
import hotgrad

class Linear():
    """
    Implements a fully connected layer
    The number of output features is the only needed parameter for initialization
    The number of input features is automatically computed during the forward pass
    """
    
    def __init__(self, output_features):
        self.output_features = output_features
        self.bias = hotgrad.variable.Variable(np.random.normal(0, 0.1, (1, output_features)), requires_grad=True)
        self.weight = None
        
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        if self.weight == None:
            input_features = input.shape[1]
            self.input_features = input_features
            a = 1/(input_features**0.5)
            self.weight = hotgrad.variable.Variable(np.random.uniform(-a, a, (self.input_features, self.output_features)), requires_grad=True)

        
        return (input @ self.weight) + self.bias
        
    def params(self):
        return [self.weight, self.bias]

    def clear(self):
        self.__init__(self.output_features)