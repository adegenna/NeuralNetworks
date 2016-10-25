import numpy as np
import matplotlib.pyplot as plt
from Edge import Edge
from MultiplyGate import MultiplyGate
from AddGate import AddGate
from SigmoidGate import SigmoidGate
import copy, sys

# A circuit: it takes 5 Units (x,y,a,b,c) and outputs a single Unit
# It can also compute the gradient w.r.t. its inputs

def forwardMultInSingleOut(X,A,MGate,AGate):
    # Subroutine for forward propagating all inputs to a single output at a particular gateLevel
    # X = vector of input state units/edges
    # A = vector of unknown parameter estimate units/edges

    # Do all multiply gates first
    mgates = [];
    for i in range(0,len(X)):
        gate = MGate[i].forward(A[i],X[i]);
        mgates.append(gate);
    # Do all add gates
    addgate = AGate[0].forward(mgates[0],mgates[1]);
    for i in range(1,len(mgates)-1):
        addgate = AGate[i].forward(addgate,mgates[i+1]);
    addgate = AGate[-1].forward(A[-1],addgate);
    return addgate;

def backwardSingleOutMultIn(MGate,AGate):
    # Subroutine that backpropagates from a single output unit
    # X = single output unit
    
    # Backpropagate through all add gates first
    for gate in reversed(AGate):
        # print("Add gate, value: " + str(gate.e0.value) + ' ' + str(gate.e1.value) + ' , grad: ' + str(gate.e0.grad) + ' ' + str(gate.e1.grad)+ ', grad_utop: ' + str(gate.utop.grad));
        gate.backward();
    # Backpropagate through all multiply gates
    for gate in MGate:
        # print("Multiply gate, value: " + str(gate.e0.value) + ' ' + str(gate.e1.value) + ' , grad: ' + str(gate.e0.grad) + ' ' + str(gate.e1.grad) + ', grad_utop: ' + str(gate.utop.grad));
        gate.backward();
        # print("Multiply gate, value: " + str(gate.e0.value) + ' ' + str(gate.e1.value) + ' , grad: ' + str(gate.e0.grad) + ' ' + str(gate.e1.grad) + ', grad_utop: ' + str(gate.utop.grad));

class Circuit:
    def __init__(self,statedim,layers,outputdim):
        # Initialize add/multiply gates
        # Gate = (Layers) x (OutputDim) x (InputDim)
        dim = np.hstack([statedim,layers,outputdim]);
        self.dim = dim;
        MGate = [];
        AGate = [];
        for i in range(0,len(dim)-1):
            rows    = dim[i+1];
            cols    = dim[i];
            MGate_i = [[MultiplyGate() for jj in range(cols)] for ii in range(rows)];
            AGate_i = [[AddGate() for jj in range(cols)] for ii in range(rows)];
            MGate.append(MGate_i);
            AGate.append(AGate_i);
        self.MGate = MGate;
        self.AGate = AGate;
  
    def forward(self,X,A):
        # Forward propagate through all gates/layers
        
        self.saveLayers = []
        for i in range(0,len(self.dim)-1):      # Index through each layer
            out = [];
            for j in range(0,self.dim[i+1]):    # Index through each individual output element
                MGate = self.MGate[i][j];
                AGate = self.AGate[i][j];
                W     = A[i][j];
                forw  = forwardMultInSingleOut(X,W,MGate,AGate);
                forw.value = np.maximum(forw.value, 0.0) if (i != len(self.dim)-2) else forw.value;
                self.saveLayers.append(forw);
                out.append(forw);
            X = copy.copy(out);
        self.forw = X;
        return X;
        
    def backward(self,gradient_top,W):
        #print("********** Start Circuit Backward ************")
        for i in range(0,len(self.forw)):
            self.forw[i].grad = gradient_top[i];
            # print("Top value: " + str(self.forw[i].value))
            # print("Top grad:  " + str(self.forw[i].grad))
        count = 0;
        for i in reversed(range(0,len(self.dim)-1)):     # Index through each layer
            for j in range(0,self.dim[-1-count]):        # Index through each individual output element
                MGate = self.MGate[i][j];
                AGate = self.AGate[i][j];
                # print("(i,j): "  + str(i) + ' ' + str(j)  + ' ' +
                #       "neuron values: "  + str(MGate[0].e1.value) + ' ' + str(MGate[1].e1.value) + ' ' + 
                #       "neuron grads: "   + str(MGate[0].e1.grad)  + ' ' + str(MGate[1].e1.grad)  + ' ' +
                #       "param grads: "    + str(MGate[0].e0.grad)  + ' ' + str(MGate[1].e0.grad));
                backwardSingleOutMultIn(MGate,AGate);
                # print("(i,j): "  + str(i) + ' ' + str(j)  + ' ' +
                #       "neuron values: "  + str(MGate[0].e1.value) + ' ' + str(MGate[1].e1.value) + ' ' + 
                #       "neuron grads: "   + str(MGate[0].e1.grad)  + ' ' + str(MGate[1].e1.grad)  + ' ' +
                #       "param grads: "    + str(MGate[0].e0.grad)  + ' ' + str(MGate[1].e0.grad));
            # Set grads to zero if neuron did not fire
            if (i != 0):
                for j in self.MGate[i]:
                    for k in j:
                        if (k.e1.value == 0):
                            # print("Apply rectifier nonlinearity")
                            k.e1.grad = 0.0;
                            # print(k.e0.value,k.e0.grad,k.e1.value,k.e1.grad)
            count += 1;
        # Set gradients in W
        
        #print("********** End Circuit Backward ************")


