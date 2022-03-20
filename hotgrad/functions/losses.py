# -*- coding: utf-8 -*-
""" Implementation of the losses. """

from hotgrad.module import Module2Operands

class MSE(Module2Operands):
    """
    Computes the Mean Squared Error
    """
    def __init__(self):
        super(MSE, self).__init__()

    def __call__(self, input, target):
        return self.forward(input, target)
        
    def forward(self, input, target):
        super(MSE, self).forward(input, target)
        assert self.l_input.data.shape == self.r_input.data.shape, "input and target sizes must be equal" # for simplicity 
        
        return (self.r_input.sub(self.l_input)).pow(2).mean()
    
    def backward(self, grad):
        l_input_grad = 2*(self.l_input - self.r_input).mean() * grad
        r_input_grad = -l_input_grad
        
        self.l_input.backward(grad = l_input_grad)
        self.r_input.backward(grad = r_input_grad)
