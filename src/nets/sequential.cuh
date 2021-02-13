#pragma once 

#include <cmath>
#include <cstdio>
#include <vector>

#include "../optimizers/optimizer.hpp"
#include "../losses/loss.hpp"
#include "../utils/utils.cuh"
#include "../Tensor/tensor.cuh"
#include "../layers/layer.cuh"

class Sequential 
{
public:
    Sequential(Loss* loss_fn, Optimizer* optim);

    Tensor* forward(Tensor* input);
    void backward(Tensor* y_pred, Tensor* y_true);
    void addLayer(Layer* layer);

private:
    std::vector<Layer*> m_layers;
    Loss* m_loss_fn;
    Optimizer* m_optim;

    Tensor* m_gradients;
};