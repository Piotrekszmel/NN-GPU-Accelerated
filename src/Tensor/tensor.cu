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

Tensor::Tensor(float* data, int sizeX, int sizeY, DataType dataType)
{
    m_sizeX = sizeX;
    m_sizeY = sizeY;
    if (dataType == HOST)
    {
        if (m_sizeX && m_sizeY)
        {
            cudaMalloc((void**)&m_devData, m_sizeX * m_sizeY * sizeof(float));
            cudaMemcpy(m_devData, data, m_sizeX * m_sizeY * sizeof(float), cudaMemcpyHostToDevice);
            //add error check
        }
        else
        {
            m_devData = NULL;
        }
    }
    else
    {
        m_devData = data;
        m_sizeX = sizeX;
        m_sizeY = sizeY;
    }
}

Tensor::~Tensor()
{
    cudaFree(m_devData);
}