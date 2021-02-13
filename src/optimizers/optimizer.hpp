#pragma once

#include <cmath>
#include <cstdio>

#include "../Tensor/tensor.cuh"

class Optimizer
{
public:
    virtual void step(Layer* layer) = 0;
};