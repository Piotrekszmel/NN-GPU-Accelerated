#pragma once

#include <cmath>
#include <cstdio>

#include "../Tensor/tensor.cuh"

class Loss
{
public:
    virtual float loss(Tensor* y_pred, Tensor* y_true) = 0;
    virtual float accuracy(Tensor* y_pred, Tensor* y_true) = 0;
    virtual Tensor* backward(Tensor* y_pred, Tensor* y_true, Tensor* output) = 0;
};
