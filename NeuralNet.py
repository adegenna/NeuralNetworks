import numpy as np
import matplotlib.pyplot as plt
from Edge import Edge
from MultiplyGate import MultiplyGate
from AddGate import AddGate
from SigmoidGate import SigmoidGate
from Circuit import Circuit

class NeuralNet:
    def __init__(self,statedim,layers,outputdim):
        # statedim:  dimension of the input state data
        # outputdim: dimension of the output state data
        # layers:    l-dimensional array of number of neurons per layer

        # Initialize weight matrix with values chosen at random from uniform[-0.5,0.5]
        # W = (Layers) x (OutputDim) x (InputDim)
        dim = np.hstack([statedim,layers,outputdim]);
        W = [];
        for i in range(0,len(dim)-1):
            rows  = dim[i+1];
            cols  = dim[i] + 1;
            vals  = np.random.random([rows,cols]) - 0.5;
            W_i = [[Edge(vals[ii,jj],0) for jj in range(cols)] for ii in range(rows)];
            W.append(W_i);
        self.W = W;
        self.dim = dim;
        self.circuit = Circuit(statedim,layers,outputdim);

    def setWeights(self,W):
        self.W = [];
        WEIGHT = [];
        for i in range(0,len(self.dim)-1):
            vals = W[i];
            rows = self.dim[i+1];
            cols = self.dim[i] + 1;
            W_i = [[Edge(vals[ii,jj],0) for jj in range(cols)] for ii in range(rows)];
            WEIGHT.append(W_i);
        self.W = WEIGHT;
    
    def forward(self,X):
        self.units_out = self.circuit.forward(X,self.W);
        return self.units_out;
    
    def backward(self,labels):
        # Reset pulls on weights
        for i in self.W:
            for j in i:
                for k in j:
                    k.grad = 0.0;
        # Compute the pull based on what the circuit output was
        pull = np.zeros(len(self.units_out));
        for i in range(0,len(self.units_out)):
            if ((labels[i]==1) & (self.units_out[i].value < 1)):
                pull[i] = 1;
            elif ((labels[i]==0) & (self.units_out[i].value > 0)):
                pull[i] = -1;
            #print("LABEL: " + str(labels[i]) + ', OUTPUT: ' + str(self.units_out[i].value) + ', PULL: ' + str(pull[i]));
        self.circuit.backward(pull,self.W);
        # print("NEURON STATES:")
        # for i in self.circuit.MGate[1]:
        #     for j in i:
        #         print(j.e1.value);
        # for i in self.circuit.MGate[2]:
        #     for j in i:
        #         print(j.e1.value);
        # print("UPDATED GRADS:")
        # for i in self.W:
        #     for j in i:
        #         for k in j:
        #             print(k.grad);
        # Add regularization pull for parameters: towards zero and proportional to value
        for i in self.W:
            for j in i:
                for k in range(0,len(j)-1):
                    j[k].grad += 0*-j[k].value;


    def parameterUpdate(self):
        step_size = 0.025;
        for i in self.W:
            for j in i:
                for k in j:
                    k.value += step_size*k.grad;


    def learnFrom(self,X,labels):
        self.forward(X);
        self.backward(labels);
        self.parameterUpdate();
