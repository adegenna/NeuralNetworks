#ifndef NET_TEST_H_
#define NET_TEST_H_

#include "../perceptron/Perceptron.h"
#include "../net/NeuralNet.h"
#include "gtest/gtest.h"
#include "math.h"

class NetTest: public ::testing::Test {
 protected:
  virtual void SetUp() {
    X_.resize(3); W_.resize(4);
    double arr1[] = {1.0,2.0,-6.0};
    double arr2[] = {0.4,0.5,0.6,0.7};
    for (int i=0; i<3; i++)
      X_[i] = arr1[i];
    for (int i=0; i<4; i++)
      W_[i] = arr2[i];
    states_  = 3;
    outputs_ = 2; 
    layers_.resize(2);
    layers_[0] = 3; layers_[1] = 2;
  }
  
  std::vector<double> X_;
  std::vector<double> W_;
  int states_, outputs_;
  std::vector<int> layers_;
};


#endif
