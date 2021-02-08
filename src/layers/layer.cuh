#pragma once

#include <cmath>
#include <cstdio>

#include "../Tensor/tensor.cuh"

class Layer
{
public:
    Layer() = default;
    Tensor* getWeights();
    Tensor* getBias();
    Tensor* getDeltaWeights();
    Tensor* getDeltaBias();

    virtual Tensor* forward(Tensor* data) = 0;
    //virtual Tensor* backward(Tensor gradients) = 0;

protected:
    Tensor* m_weights;
    Tensor* m_bias;
    Tensor* m_delta_weights;
    Tensor* m_delta_bias;
};
