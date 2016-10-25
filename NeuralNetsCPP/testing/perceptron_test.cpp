#include "perceptron_test.h"

TEST_F(PerceptronTest, PerceptronForward) {
  // Test forward propagation as well as get methods
  Perceptron p(W_);
  EXPECT_FLOAT_EQ(3.9, p.forward(X_));
  std::vector<double> states  = p.getInput();
  std::vector<double> weights = p.getWeights();
  for (int i=0; i<3; i++)
    EXPECT_FLOAT_EQ(i+1.0, states[i]);
  for (int i=0; i<4; i++)
    EXPECT_FLOAT_EQ(0.4 + i*0.1, weights[i]);
  
}

TEST_F(PerceptronTest, PerceptronBackward) {
  // Test backward propagation
  Perceptron p(W_);
  double G = 1.0;
  p.forward(X_);
  p.backward(G);
  std::vector<double> gradW = p.getGradW();
  std::vector<double> gradS = p.getGradS();
  for (int i=0; i<3; i++) {
    EXPECT_FLOAT_EQ(G*W_[i] , gradS[i]);
    EXPECT_FLOAT_EQ(G*X_[i] , gradW[i]);
  }
  EXPECT_FLOAT_EQ(G , gradW.back());
  
}
