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

# http://karpathy.github.io/neuralnets/

# Driver program
data = []; labels = [];
data.append(np.array([ 1.2 , 0.7 ])); labels.append(1);
data.append(np.array([-0.3 , -0.5])); labels.append(-1);
data.append(np.array([ 3.0 , 0.1 ])); labels.append(1);
data.append(np.array([-0.1 , -1.0])); labels.append(-1);
data.append(np.array([-1.0 , 1.1 ])); labels.append(-1);
data.append(np.array([ 2.1 , -3.0])); labels.append(1);
svm = SVM();
# Function that computes the classification accuracy
def evalTrainingAccuracy():
    num_correct = 0.0;
    for i in range(0,len(data)):
        x          = Edge(data[i][0],0.0);
        y          = Edge(data[i][1],0.0);
        true_label = labels[i];
        # See if prediction matches the provided label
        predicted_label = 1 if svm.forward(x,y).value > 0 else -1;
        if (predicted_label == true_label):
            num_correct += 1.0;
    return num_correct/len(data);

# Learning loop
for iter in range(0,1000):
    # Pick a random data point
    i     = int(np.floor(np.random.random() * len(data)));
    x     = Edge(data[i][0],0.0);
    y     = Edge(data[i][1],0.0);
    label = labels[i];
    svm.learnFrom(x,y,label);
    accur = evalTrainingAccuracy();
    if (accur > 0.99):
        break;
    if (np.mod(iter,25) == 0):
        print("Training accuracy at iteration " + str(iter) + ": " + str(accur) + " , a = " + str(svm.a.value) + ", b = " + str(svm.b.value) + ", c = " + str(svm.c.value));
        plt.scatter(iter,svm.a.value,s=20,c='b')
        plt.scatter(iter,svm.b.value,s=20,c='r');
        plt.scatter(iter,svm.c.value,s=20,c='g');

for i in range(0,len(data)):
    x     = Edge(data[i][0],0.0);
    y     = Edge(data[i][1],0.0);
    label = labels[i];
    pred = svm.circuit.forward(x,y,svm.a,svm.b,svm.c).value;
    print("Prediction = " + str(pred) + ", Actual = " + str(label));
plt.show()
