#include "net_test.h"

TEST_F(NetTest, NetInitialize) {
  // Test network initialization
  NeuralNet net(states_,layers_,outputs_);
  int states  = net.getStateDim();
  int outputs = net.getOutputDim();
  std::vector<int> layers = net.getLayerDim();
  EXPECT_FLOAT_EQ(states , states_);
  EXPECT_FLOAT_EQ(outputs , outputs_);
  EXPECT_FLOAT_EQ(3 , layers_[0]);
  EXPECT_FLOAT_EQ(2 , layers_[1]);
  std::vector<std::vector<Perceptron> > perceptrons = net.getPerceptrons();
  EXPECT_EQ(perceptrons.size() , layers_.size() + 1);
  EXPECT_EQ(perceptrons[0].size() , layers_[0]);
  EXPECT_EQ(perceptrons[1].size() , layers_[1]);
  EXPECT_EQ(perceptrons[2].size() , outputs_);
  
}

TEST_F(NetTest, NetWeights) {
  // Test ability to set random network weights
  NeuralNet net(states_,layers_,outputs_);
  double lower = -1.0; double upper = 1.0;
  int states  = net.getStateDim();
  int outputs = net.getOutputDim();
  std::vector<int> layers = net.getLayerDim();
  net.setRandomWeights(lower,upper);
  std::vector<std::vector<Perceptron> > perceptrons = net.getPerceptrons();
  std::vector<double> weights;
  for (int i=0; i<layers.size(); i++) {
    for (int j=0; j<layers[i]; j++) {
      weights = perceptrons[i][j].getWeights();
      for (int k=0; k<weights.size(); k++) {
  	EXPECT_GE(weights[k] , lower);
  	EXPECT_LE(weights[k] , upper);
      }
    }
    weights.clear();
  }
  
}

TEST_F(NetTest, NetForward) {
  // Test ability to forward step network
  NeuralNet net(states_,layers_,outputs_);
  double lower = -1.0; double upper = 1.0;
  net.setUnitWeights();
  std::vector<double> output;
  output = net.forward(X_);
  // Compute forward network manually as test
  std::vector<std::vector<Perceptron> > perceptrons = net.getPerceptrons();
  std::vector<double> weights;
  std::vector<double> Xnew,X;
  X = X_;
  for (int i=0; i<perceptrons.size(); i++) {
    Xnew.assign(perceptrons[i].size() , 0.0);
    for (int j=0; j<perceptrons[i].size(); j++) {
      weights = perceptrons[i][j].getWeights();
      for (int k=0; k<weights.size()-1; k++) {
	Xnew[j] += weights[k]*X[k];
      }
      Xnew[j] += weights.back();
      if ((Xnew[j] < 0) && (i != perceptrons.size()-1))
	Xnew[j] = 0.0; // No neuron can have a value less than 0
      std::cout << '(' << i << ',' << j << ')' << ": " << Xnew[j] << '\n';
    }
    std::cout << '\n';
    X = Xnew;
    weights.clear();
  }
  EXPECT_EQ(X[0] , output[0]);
  EXPECT_EQ(X[1] , output[1]);
    
}

TEST_F(NetTest, NetBackward) {
  // Test backpropagation of gradients through network
  NeuralNet net(states_,layers_,outputs_);
  net.setUnitWeights();
  std::vector<double> output;
  output = net.forward(X_);
  std::vector<double> G(2); G[0] = 1.0; G[1] = 1.0;
  net.backward(G);
  // Compute backpropagation manually as test
  std::vector<std::vector<Perceptron> > perceptrons = net.getPerceptrons();
  std::vector<double> gradW,gradS;
  std::vector<double> gradWchecks(2); gradWchecks[1] = 2.0;  gradWchecks[0] = 0.0;
  std::vector<double> gradSchecks(2); gradSchecks[1] = 2.0;  gradSchecks[0] = 0.0;
  
  for (int j=0; j<perceptrons[0].size(); j++) {
    gradW = perceptrons[0][j].getGradW(); gradS = perceptrons[0][j].getGradS();
    for (int k=0; k<gradS.size(); k++) 
      std::cout << '(' << '0' << ',' << j << ',' << k << ')' << ": " << gradW[k] << " , " << gradS[k] << '\n';
    std::cout << '(' << '0' << ',' << j << ',' << gradW.size()-1 << ')' << ": " << gradW.back() << " , " << gradS.back() << '\n';
    EXPECT_EQ(0,gradW[0]); EXPECT_EQ(0,gradS[0]); 
    EXPECT_EQ(0,gradW[1]); EXPECT_EQ(0,gradS[1]);
    EXPECT_EQ(0,gradW[2]); EXPECT_EQ(0,gradS[2]);
    EXPECT_EQ(0,gradW[3]);
  }
  std::cout << '\n';
  for (int i=1; i<perceptrons.size(); i++) {
    for (int j=0; j<perceptrons[i].size(); j++) {
      gradW = perceptrons[i][j].getGradW();
      gradS = perceptrons[i][j].getGradS();
      for (int k=0; k<gradS.size(); k++) {
	std::cout << '(' << i << ',' << j << ',' << k << ')' << ": " << gradW[k] << " , " << gradS[k] << '\n';
	EXPECT_EQ(gradW[k] , gradWchecks[i-1]); EXPECT_EQ(gradS[k] , gradSchecks[i-1]); 
      }
      EXPECT_EQ(gradW.back() , -2.0*i + 6.0);
      std::cout << '(' << i << ',' << j << ',' << gradW.size()-1 << ')' << ": " << gradW.back() << '\n';
    }
    std::cout << '\n';
  }

}
