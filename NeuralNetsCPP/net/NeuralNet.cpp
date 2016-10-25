#include "NeuralNet.h"
#include <iostream>
#include <chrono>
#include <random>
#include <assert.h>
#include "../vectoroperations/VectorOperations.h"

NeuralNet::NeuralNet(int states, std::vector<int>& layers, int outputs)
  : states_(states) , layers_(layers) , outputs_(outputs) {
  // Constructor
  int numLayers      = layers.size();
  int numConnections = numLayers + 1;
  perceptrons_.resize(numConnections);
  for (int i=0; i<numLayers; i++) {
    perceptrons_[i].resize(layers_[i]);
  }
  perceptrons_[numLayers].resize(outputs_);
  // Set number of states for each perceptron
  for (int i=0; i<perceptrons_[0].size(); i++)
    perceptrons_[0][i].setNumStates(states_);
  for (int i=0; i<numLayers; i++) {
    for (int j=0; j<perceptrons_[i+1].size(); j++)
      perceptrons_[i+1][j].setNumStates(layers_[i]);
  }
}

NeuralNet::~NeuralNet() { }

int NeuralNet::getStateDim()  {  return states_;  }
int NeuralNet::getOutputDim() {  return outputs_;  }
std::vector<int> NeuralNet::getLayerDim()  {  return layers_;  }
std::vector<std::vector<Perceptron> > NeuralNet::getPerceptrons()  {  return perceptrons_;  }

void NeuralNet::setRandomWeights(double lower, double upper) {
  // Function to set random weights of network perceptrons

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> weightDistribution(lower,upper);
  int numStates = 0;
  std::vector<double> weights;
  for (int i=0; i<perceptrons_.size(); i++) {       // Index over each connection layer
    for (int j=0; j<perceptrons_[i].size(); j++) {  // Index over each individual perceptron in a given layer
      numStates = perceptrons_[i][j].getNumStates();
      weights.resize(numStates+1);
      for (int k=0; k<weights.size(); k++)             // Index over each weight for a given perceptron
  	weights[k] = weightDistribution(generator);
      perceptrons_[i][j].setWeights(weights);
    }
  }

}

void NeuralNet::setUnitWeights() {
  // Function to set all weights of network perceptrons to 1

  int numStates = 0;
  std::vector<double> weights;
  for (int i=0; i<perceptrons_.size(); i++) {       // Index over each connection layer
    for (int j=0; j<perceptrons_[i].size(); j++) {  // Index over each individual perceptron in a given layer
      numStates = perceptrons_[i][j].getNumStates();
      weights.resize(numStates+1);
      for (int k=0; k<weights.size(); k++)             // Index over each weight for a given perceptron
  	weights[k] = 1.0;
      perceptrons_[i][j].setWeights(weights);
    }
  }

}


std::vector<double> NeuralNet::forward(std::vector<double> X) {
  // Function to step network forward

  std::vector<double> Xnew(X.size());
  double output = 0.0;
  for (int i=0; i<perceptrons_.size(); i++) {
    Xnew.resize(perceptrons_[i].size());
    for (int j=0; j<perceptrons_[i].size(); j++) {
      output  = perceptrons_[i][j].forward(X);
      if ((output < 0) && (i != perceptrons_.size()-1))
	output = 0.0; // No neuron can have a value less than 0
      Xnew[j] = output;
    }
    X.resize(Xnew.size());
    X = Xnew;
  }
  return X;

}

void NeuralNet::backward(std::vector<double>& G) {
  // Function to backpropagate gradient through the network

  std::vector<double> gradW, gradS, states;
  int inputsize;
  for (int i=perceptrons_.size()-1; i>-1; i--) {
    inputsize = perceptrons_[i][0].getNumStates();
    gradW.assign(inputsize+1, 0.0);
    gradS.assign(inputsize  , 0.0);
    // Calculate/set new weight/state gradients for layer of neurons
    assert(perceptrons_[i].size() == G.size());
    for (int j=0; j<perceptrons_[i].size(); j++) {
      perceptrons_[i][j].backward(G[j]);
      gradW = gradW + perceptrons_[i][j].getGradW();
      gradS = gradS + perceptrons_[i][j].getGradS();
    }
    for (int j=0; j<perceptrons_[i].size(); j++) {
      states = perceptrons_[i][j].getInput();
      for (int k=0; k<gradS.size(); k++) {
      	if ((states[k] == 0.0) && (i != perceptrons_.size()-1))
      	  gradS[k] = 0.0; // Set to zero if neuron did not fire
      }
      perceptrons_[i][j].setGradW(gradW);
      perceptrons_[i][j].setGradS(gradS);
    }
    // Reset backpropagated gradients for next layer
    G = gradS;
  }
}

void NeuralNet::parameterUpdate() {
  // Update parameters in netrork

  double step_size = 0.01;
  


}
