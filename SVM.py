import numpy as np
import matplotlib.pyplot as plt
from Edge import Edge
from MultiplyGate import MultiplyGate
from AddGate import AddGate
from SigmoidGate import SigmoidGate
from Circuit import Circuit

class SVM:
    def __init__(self):
        # Random initial parameter values
        self.a = Edge(1.0,0.0);
        self.b = Edge(-2.0,0.0);
        self.c = Edge(-1.0,0.0);
        self.circuit = Circuit();
    
    def forward(self,x,y):
        self.edge_out = self.circuit.forward(x,y,self.a,self.b,self.c);
        return self.edge_out;
    
    def backward(self,label):
        # Reset pulls on a,b,c
        self.a.grad = 0.0;
        self.b.grad = 0.0;
        self.c.grad = 0.0;
        # Compute the pull based on what the circuit output was
        pull = 0.0;
        if ((label==1) & (self.edge_out.value < 1)):
            pull = 1;
        elif ((label==-1) & (self.edge_out.value > -1)):
            pull = -1;
        self.circuit.backward(pull);
        # Add regularization pull for parameters: towards zero and proportional to value
        self.a.grad += -self.a.value;
        self.b.grad += -self.b.value;

    def parameterUpdate(self):
        step_size = 0.01;
        self.a.value += step_size*self.a.grad;
        self.b.value += step_size*self.b.grad;
        self.c.value += step_size*self.c.grad;

    def learnFrom(self,x,y,label):
        self.forward(x,y);
        self.backward(label);
        self.parameterUpdate();
