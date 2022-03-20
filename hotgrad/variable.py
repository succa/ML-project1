# -*- coding: utf-8 -*-
""" The Variable object stores the previous operation so to allow the gradient
to packpropagate to the previous operation. """

#from torch import FloatTensor
import numpy as np

from hotgrad.module import Module

from hotgrad.functions.operators import *
from hotgrad.functions.losses import *
from hotgrad.functions.activations import ReLU, Tanh
from hotgrad.exceptions import BackwardException


# For now assume all passed parameters are of class Variable (except for the pow())
class Variable():
    """ Variable are the basic """
    def __init__(self, data, previous_op=None, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array([data])
        assert isinstance(data, np.ndarray), "The data stored in a Variable must be a pytorch Tensor"
        assert (isinstance(previous_op, Module)) or (previous_op is None), "The operation that created this Variable is not a valid Module"
        assert isinstance(requires_grad, bool), "The parameter requires_grad must be a boolean."
        
        self.data = data
        self.shape = data.shape
        self.previous_op = previous_op
        self.requires_grad = requires_grad
        self.grad = np.empty(data.shape)
        self.grad.fill(0)

    def __getitem__(self, indexes):
        assert(self.previous_op == None), "previous_op must be None"
        
        variable = Variable(self.data[indexes], previous_op=self.previous_op, requires_grad=self.requires_grad)
        variable.grad = self.grad[indexes]
        return variable

    def __mul__(self, other):
        """ Multiplies this Variable with either another Variable (element-wise by 
        broadcasting if necessary) or a constant, i.e. 'other' can be of type 
        Variable or int/float."""
        return Mul().forward(self, other)
    
    def __truediv__(self, other):
        """ Divides this Variable with either another Variable (element-wise by 
        broadcasting if necessary) or a constant, i.e. 'other' can be of type 
        Variable or int/float."""
        print("div")
        return 
    
    def __sub__(self, other):
        """ Subtracts this Variable with either another Variable (element-wise by 
        broadcasting if necessary) or a constant, i.e. 'other' can be of type 
        Variable or int/float."""
        return Sub().forward(self, other)
    
    def __add__(self, other):
        """ Add this Variable with either another Variable (element-wise by 
        broadcasting if necessary) or a constant, i.e. 'other' can be of type 
        Variable or int/float."""
        return Add().forward(self, other)
    
    def __pow__(self, other):
        """ Compute the power of this Variable (element-wise) by a constant, 
        i.e. 'other' can only be of type int/float."""
        return Pow().forward(self, other)
    
    def __matmul__(self, other):
        """ Multiplies this Variable by another Variable, i.e. 'other' can only 
        be of type Variable and its shape has to allow for matric multiplication."""
        return MatMul().forward(self, other)
    
    def mean(self):
        return Mean().forward(self)
    
    def relu(self):
        return ReLU().forward(self)
    
    def tanh(self):
        return Tanh().forward(self)
    
    def pow(self, other):
        return self.__pow__(other)
    
    def add(self, other):
        return self.__add__(other)
    
    def sub(self, other):
        return self.__sub__(other)
    
    #def backward(self, grad=FloatTensor([1])):
    def backward(self, grad=np.array([1])):
        # if the backpropagation starts here then shape of this Variable must be (1,)
        # (the gradient can be computed implicitly only for scalar output)

        if len(grad.shape) != len(self.data.shape):
            raise BackwardException("The number of dimensions of the received gradient is not equal to the number of dimensions of this Variable.")
        if self.data.shape[0] != 1 and grad.shape[0] != self.data.shape[0]:
            raise BackwardException("The first dimension of this Variable must be either 1 or equal to the first dimension of the received gradient. Grad shape:" + str(grad.shape) + ". Variable shape:" + str(self.data.shape))
        if len(grad.shape)>1 and self.data.shape[1] != 1 and grad.shape[1] != self.data.shape[1]:
            raise BackwardException("The second dimension of this Variable must be either 1 or equal to the second dimension of the received gradient. Grad shape: " + str(grad.shape) + ". Variable shape: " + str(self.data.shape))
        
        # sum over the dimensions if broadcasting was used
        if grad.shape[0] != self.data.shape[0]:
            grad = np.expand_dims(grad.sum(0), axis=0)
        if len(grad.shape)>1 and (grad.shape[1] != self.data.shape[1]):
            grad = np.expand_dims(grad.sum(1), axis=1)
            
        # check if this variable requires the gradient. If so then update it's local gradient.
        if (self.requires_grad and grad is not None):            
            self.grad += grad
            
        # finally propagate the gradient
        if (self.previous_op is not None):
            self.previous_op.backward(grad) # propagate the gradient
            
    def zero_grad(self):
        if self.requires_grad:
            self.grad.fill(0)
            
    def __str__(self):
        return "Variable containing:" + str(self.data)
    
    def __repr__(self):
        return self.__str__()
