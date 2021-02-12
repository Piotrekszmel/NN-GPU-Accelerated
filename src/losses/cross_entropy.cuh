#pragma once

#include <cmath>
#include <cstdio>

#include "../Tensor/tensor.cuh"
#include "../config/config.cuh"
#include "loss.hpp"

class CrossEntropy : public Loss 
{
public:
    float loss(Tensor* y_pred, Tensor* y_true);
    float accuracy(Tensor* y_pred, Tensor* y_true);
    Tensor* backward(Tensor* y_pred, Tensor* y_true, Tensor* gradients);
};
