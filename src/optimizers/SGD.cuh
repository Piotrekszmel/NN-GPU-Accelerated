#pragma once 

#include <cmath>
#include <cstdio>

#include "../Tensor/tensor.cuh"
#include "../layers/layer.cuh"
#include "optimizer.hpp"

class SGD : public Optimizer
{
public:
    SGD(float lr);

    void step(Layer* layer);
private:
    float m_lr;
};
