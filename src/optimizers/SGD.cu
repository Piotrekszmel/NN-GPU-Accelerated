#include "SGD.cuh"

SGD::SGD(float lr)
{
    m_lr = lr;
}

void SGD::step(Layer* layer)
{
    if (layer->getDeltaWeights())
    {
        layer->getDeltaWeights()->scale(m_lr);
    }
    if (layer->getDeltaBias())
    {
        layer->getDeltaBias()->scale(m_lr);
    }

    if (layer->getWeights())
    {
        layer->getWeights()->subtract(layer->getDeltaWeights());
    }
    if (layer->getBias())
    {
        layer->getBias()->subtract(layer->getDeltaBias());
    }
}