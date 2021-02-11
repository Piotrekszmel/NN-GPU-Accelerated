#pragma once

#include <cmath>
#include <cstdio>

#include "../Tensor/tensor.cuh"
#include "../config/config.cuh"

class ReLU
{
public:
    ReLU();

    Tensor* activation(Tensor* data);
    Tensor* derivative(Tensor* gradients);
private:
    Tensor* m_a;
    Tensor* m_gradients;
};