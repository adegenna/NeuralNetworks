// Implementation of the perceptron structure

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <stdlib.h>
#include <vector>
#include "../perceptron/Perceptron.h"

class NeuralNet {
  public:
    NeuralNet(int states, std::vector<int>& layers, int outputs);
    ~NeuralNet();
    int getStateDim();
    int getOutputDim();
    std::vector<int> getLayerDim();
    std::vector<std::vector<Perceptron> > getPerceptrons();
    void setRandomWeights(double lower, double upper);
    void setUnitWeights();
    std::vector<double> forward(std::vector<double> X);
    void backward(std::vector<double>& G);
    void parameterUpdate();
    
  private:
    int states_;
    int outputs_;
    std::vector<int> layers_;
    std::vector<std::vector<Perceptron> > perceptrons_;
    std::vector<double> stateValues_;
    std::vector<double> outputValues_;
};

#endif
