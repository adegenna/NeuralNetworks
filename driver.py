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

# Written by Anthony DeGennaro
# Basic ideas/tutorials from http://karpathy.github.io/neuralnets/

# Function that computes the classification accuracy
def evalTrainingAccuracy():
    num_correct = 0.0;
    for i in range(0,len(data)):
        inp   = data[i][0];
        x     = [Edge(j,0.0) for j in inp];
        true_label = data[i][1];
        # See if prediction matches the provided label
        vals = net.forward(x);
        num = np.size(vals);
        for j in range(num):
            val = vals[j].value;
            predicted_label = 1 if val > 0 else -1;
            if (predicted_label == true_label[j]):
                num_correct += 1.0/num;
    return num_correct/len(data);
    

# ********************************
# INPUT DATA
# ********************************
# data = []; labels = [];
# numdata = 500;
# for i in range(0,numdata):
#     xval = np.random.uniform(-5,5)
#     yval = np.random.uniform(-5,5);
#     data.append(np.array([xval,yval]));
#     label = np.array([0,0]);
#     #if (((xval > 0) & (yval > 0)) | ((xval<0) & (yval<0)) ):
#     #if (np.power(xval,2)+np.power(yval,2) - 9.0 > 0):
#     if (np.power(xval,2) + np.power(yval,2) > 9.0):
#         label[0] = 1;
#     else:
#         label[0] = -1;
#     if (xval-yval > 0):
#         label[1] = 1;
#     else:
#         label[1] = -1;
#     labels.append(label);

# ********************************
# NETWORK INITIALIZATION
# ********************************
net  = NeuralNet(784,15,10);
data = mnist.load_data_wrapper();
data = data[0];

# ********************************
# LEARNING LOOP
# ********************************
outMonitor = open("/home/adegenna/NeuralNetworks/OUT.out","a");
for iter in range(0,50000):
    outMonitor.write(str(iter) + "\n");
    # Pick a random data point
    #i     = int(np.floor(np.random.random() * len(data)));
    i     = int(iter);
    inp   = data[i][0];
    x     = [Edge(j,0.0) for j in inp]; 
    label = data[i][1];
    net.learnFrom(x,label);
    # if (np.mod(iter+1,100)==0):
    #     accur = evalTrainingAccuracy();
    #     print("Training accuracy at iteration " + str(iter) + ": " + str(accur));
    #     if (accur > 0.95):
    #         print("Training accuracy at iteration " + str(iter) + ": " + str(accur));
    #         break;
outMonitor.close()
# Output model weights to file
count = 1;
for i in net.W:
    outfile = open("/home/adegenna/NeuralNetworks/WEIGHTS" + str(count) + ".csv","a");
    for j in i:
        for k in j:
            outfile.write(str(float(k.value)) + ', ');
        outfile.write('\n');
    outfile.write('\n');
    outfile.close();
    count += 1;

# *******************************
# DISPLAY RESULTS
# *******************************
# xval,yval = np.meshgrid(np.linspace(-5,5,50),np.linspace(-5,5,50));
# xval = xval.reshape(2500); yval = yval.reshape(2500)
# for i in range(0,2500):
#     x    = Edge(xval[i],0.0);
#     y    = Edge(yval[i],0.0);
#     val1  = net.forward([x,y])[0].value;
#     val2  = net.forward([x,y])[1].value;
#     col1  = 'r' if val1 > 0 else 'b';
#     col2  = 'r' if val2 > 0 else 'b';
#     plt.subplot(211); plt.scatter(xval[i],yval[i],s=20,c=col1);
#     plt.subplot(212); plt.scatter(xval[i],yval[i],s=20,c=col2);
# for i in range(0,numdata):
#     col1 = 'g' if labels[i][0] == 1 else 'none';
#     col2 = 'g' if labels[i][1] == 1 else 'none';
#     plt.subplot(211); plt.scatter(data[i][0],data[i][1],s=20,c=col1);
#     plt.subplot(212); plt.scatter(data[i][0],data[i][1],s=20,c=col2);
# plt.gca().set_aspect('equal');

# plt.show()
