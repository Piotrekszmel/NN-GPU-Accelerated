#include "tensor.cuh"

Tensor::Tensor(int sizeX, int sizeY)
{
    m_sizeX = sizeX;
    m_sizeY = sizeY;

    if (m_sizeX && m_sizeY)
    {
        cudaMalloc((void**)&m_devData, m_sizeX * m_sizeY * sizeof(float));
    }
    else
    {
        m_devData = NULL;
    }
}