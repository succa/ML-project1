# -*- coding: utf-8 -*-
""" Implementation of the "abstract" class representing the module."""

class Module(object):
    """ Base class that provides the interface that all the modules should implement. Each module
    is part of the newtwork and, other that implementing the operation on the input (forward pass),
    has to implement the backward step so to propagate the gradient to the previous modules in the
    network.
    """
    
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def params(self):
        return []
    
class Module2Operands(Module):
    """ Base class for modules accepting 2 operands in the forward pass. """
    def __init__(self):
        self.l_input = None
        self.r_input = None
    
    def __str__(self):
        return (self.__class__.__name__ + " operator:\n" + 
                "\n------ Left operand ------\n" + str(self.l_input) + "\n--------------------------\n" +
                "\n------ Right operand ------\n" + str(self.r_input) + "\n--------------------------")
    
    def __repr__(self):
        return self.__str__()
    
    def __call__(self, l_input, r_input):
        return self.forward(l_input, r_input)
    
    def forward(self, l_input, r_input):
        """ Computes, after storing a pointer to the operands, the forward pass. """
        self.l_input = l_input
        self.r_input = r_input

class Module1Operand(Module):
    """ Base class for modules accepting only one operand as input in the forward pass. """
    def __init__(self):
        self.input = None
        
    def __str__(self):
        return (self.__class__.__name__ + " operator:\n" + 
                "\n------ Operand ------\n" + str(self.input) + "\n--------------------------\n")
    
    def __repr__(self):
        return self.__str__()

    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        """ Computes, after storing a pointer to the operand, the forward pass. """
        self.input = input
