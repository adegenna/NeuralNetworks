#include "Perceptron.h"
#include <cmath>

Perceptron::Perceptron() { }

Perceptron::Perceptron(std::vector<double> W)
  : W_(W) , numStates_(W.size()-1) {
  gradW_.assign(numStates_+1 , 0.0);
  gradS_.assign(numStates_   , 0.0);
  Sin_.assign(numStates_     , 0.0);
}

Perceptron::~Perceptron() { }

double Perceptron::forward(std::vector<double> X) {
  // Compute and set perceptron output
  double Y = 0.0;
  for (int i=0; i<numStates_; i++) {
    Y      += W_[i]*X[i];
    Sin_[i] = X[i];
  }
  Y    += W_.back();
  Sout_ = Y;
  return Y;
}

void Perceptron::backward(double G) {
  // Compute backpropagated input state/weight gradients
  for (int i=0; i<numStates_; i++) {
    gradW_[i] += G*Sin_[i];
    gradS_[i] += G*W_[i];
  }
  gradW_[numStates_] += G;
  
}

double Perceptron::getOutput()                        {  return Sout_;   }
std::vector<double> Perceptron::getInput()            {  return Sin_;  }
std::vector<double> Perceptron::getWeights()          {  return W_;   }
std::vector<double> Perceptron::getGradW()            {  return gradW_;  }
std::vector<double> Perceptron::getGradS()            {  return gradS_;   }
int Perceptron::getNumStates()                       {  return numStates_;   }
void Perceptron::setNumStates(int states)            {  numStates_ = states;  }
void Perceptron::setGradW(std::vector<double>& gradW) {  gradW_ = gradW;  }
void Perceptron::setGradS(std::vector<double>& gradS) {  gradS_ = gradS;  }

void Perceptron::setWeights(std::vector<double>& weights) {
  W_.assign(numStates_+1     , 0.0);
  Sin_.assign(numStates_     , 0.0);
  gradW_.assign(numStates_+1 , 0.0);
  gradS_.assign(numStates_   , 0.0);
  for (int i=0; i<numStates_+1; i++)
    W_[i] = weights[i];
}


