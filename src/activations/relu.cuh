#pragma once

#include <cmath>
#include <cstdio>

#include "../Tensor/tensor.cuh"
#include "../config/config.cuh"
#include "../layers/layer.cuh"

class ReLU : public Layer
{
public:
    ReLU();

    Tensor* forward(Tensor* data);
    Tensor* backward(Tensor* gradients);
private:
    Tensor* m_a;
    Tensor* m_gradients;
};