#include "layer.cuh"

Tensor* Layer::getWeights()
{
    return this->m_weights;
}

Tensor* Layer::getBias()
{
    return this->m_bias;
}

Tensor* Layer::getDeltaWeights()
{
    return this->m_delta_weights;
}

Tensor* Layer::getDeltaBias()
{
    return this->m_delta_bias;
}
