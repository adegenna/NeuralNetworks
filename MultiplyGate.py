import numpy as np
from Edge import Edge

class MultiplyGate:
    def __init__(self):
        self.e0   = 0.0;
        self.e1   = 0.0;
        self.utop = 0.0;

    def forward(self, e0, e1):
        self.e0   = e0;
        self.e1   = e1;
        self.utop = Edge(e0.value * e1.value, 0.0);
        return self.utop;
    
    def backward(self):
        self.e0.grad += self.e1.value * self.utop.grad;
        self.e1.grad += self.e0.value * self.utop.grad;
