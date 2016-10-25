/**
 * \file
 *
 * \brief Driver program for testing of sph
 */
#include <stdlib.h>
#include <iostream>
#include "gtest/gtest.h"

using namespace std;

int main(int argc, char **argv) {

  cout << "Testing perceptron properties: " << endl;

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

}
