import numpy as np
from Edge import Edge

def sigmoid(e):
    x = e.value;
    return 1.0/(1.0 + np.exp(-x));

class SigmoidGate:
    def __init__(self):
        self.e0   = 0.0;
        self.e1   = 0.0;
        self.utop = 0.0;

    def forward(self, e0):
        self.e0   = e0;
        self.utop = Edge(sigmoid(e0) , 0.0);
        return self.utop;
    
    def backward(self):
        s = sigmoid(self.e0);
        self.e0.grad += s*(1-s) * self.utop.grad;
