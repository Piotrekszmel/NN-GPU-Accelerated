#include "layer.cuh"

Tensor* Layer::getWeights()
{
    return m_weights;
}

Tensor* Layer::getBias()
{
    return m_bias;
}

Tensor* Layer::getDeltaWeights()
{
    return m_delta_weights;
}

Tensor* Layer::getDeltaBias()
{
    return m_delta_bias;
}
