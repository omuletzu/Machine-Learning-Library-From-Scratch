cmake_minimum_required(VERSION 3.26)
project(ML_lib)

set(CMAKE_CXX_STANDARD 17)

add_executable(ML_lib main.cpp
        Matrix.h
        Matrix.cpp
        ML.h
        ML.cpp
        Layer.cpp
        Layer.h
        Node.cpp
        Node.h)
