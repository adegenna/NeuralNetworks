cmake_minimum_required(VERSION 2.6)
project(neuralnet)

find_package(Boost)
find_package(GTest REQUIRED)
find_package(Threads)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-w -O3 -g -std=c++11 -DNDEBUG ")
set(CMAKE_CXX_FLAGS_DEBUG "-w -O0 -g -std=c++11 ")

# sets a preprocessor macro to allow relative test case input directories
add_definitions(-DTESTCASEDIR=\"${CMAKE_SOURCE_DIR}/TESTCASES/\")
add_definitions( -std=c++11 -c -g )

include_directories(
        ${Boost_INCLUDE_DIR}
        ${GTEST_INCLUDE_DIRS}
        "/usr/include/eigen3")

set(NEURALNET_SRC
        perceptron/Perceptron.cpp perceptron/Perceptron.h
	net/NeuralNet.cpp net/NeuralNet.h
	vectoroperations/VectorOperations.h
) 

set(TEST_SRC
        ${NEURALNET_SRC}
        testing/perceptron_test.cpp testing/perceptron_test.h
	testing/net_test.cpp testing/net_test.h
	vectoroperations/VectorOperations.h
)

# add_executable(neuralnet neuralnet.cpp ${NEURALNET_SRC})

add_executable(tests testing/tests.cc ${TEST_SRC})
target_link_libraries(tests
        ${GTEST_LIBRARIES}
        ${Boost_LIBRARIES}
        /usr/lib/libgtest.a
        pthread
        )