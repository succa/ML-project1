# -*- coding: utf-8 -*-

""" Implementation of the optimizers. """

class SGD():
    def __init__(self, lr = 0.01):
        self.lr = lr

    def set_params(self, params):
        self.parameters = params
        
    def params(self):
        return self.parameters
        
    """ updates all the inputs of the modules with their gradient """
    def step(self):
        for param in self.params():
            param.data -= (self.lr * param.grad)