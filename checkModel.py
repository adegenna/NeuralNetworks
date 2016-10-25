import numpy as np
import matplotlib.pyplot as plt
from Edge import Edge
from MultiplyGate import MultiplyGate
from AddGate import AddGate
import numpy as np
import matplotlib.pyplot as plt
from Edge import Edge
from MultiplyGate import MultiplyGate
from AddGate import AddGate
from SigmoidGate import SigmoidGate
from Circuit import Circuit
from SVM import SVM
from NeuralNet import NeuralNet
import sys
import mnist_loader as mnist

W1 = np.genfromtxt('WEIGHTS1.csv',delimiter=',');
W2 = np.genfromtxt('WEIGHTS2.csv',delimiter=',');
W1 = W1[0:15,0:785];
W2 = W2[0:10,0:16];
W = [W1,W2];
net  = NeuralNet(784,15,10);
data = mnist.load_data_wrapper();
net.setWeights(W);
count = 0.0;
for i in range(0,10000):
    xval = data[0][i][0];
    x     = [Edge(j,0.0) for j in xval];
    label = data[0][i][1];
    y = net.forward(x);
    num = np.size(y);
    predicted_label = np.zeros(num);
    for j in range(num):
        val = y[j].value;
        predicted_label[j] = 0 if val<0 else 1
    label = np.where(label==1)[0][0]
    if ((predicted_label[label] == 1) ):
        count += 1;
    print(i,predicted_label,label)
            
print(count/10000);
