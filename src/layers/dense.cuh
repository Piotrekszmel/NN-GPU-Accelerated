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
    Tensor forward(Tensor& data);
    Tensor backward(Tensor& gradients);

private:
    Tensor in_data;
    Tensor out_data;
    Tensor out_backward;

    int m_input_size;
    int m_output_size;
};
