#pragma once

#include <cmath>
#include <cstdio>

#include "../Tensor/tensor.cuh"
#include "../utils/utils.cuh"
#include "layer.cuh"

class Dense : public Layer 
{
public:
    Dense(int input_size, int output_size);
    Tensor* forward(Tensor* data);
    Tensor* backward(Tensor* gradients);

private:
    Tensor* m_in_data;
    Tensor* m_z;
    Tensor* m_gradients;

    int m_input_size;
    int m_output_size;
};
