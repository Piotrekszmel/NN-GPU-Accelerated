#include "tensor.cuh"
#include "../config/config.cuh"

/* CONSTRUCTORS */ 

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
    if (dataType == HostToDevice)
    {
        if (m_sizeX && m_sizeY)
        {
            gpuErrCheck(cudaMalloc((void**)&m_devData, m_sizeX * m_sizeY * sizeof(float)));
            gpuErrCheck(cudaMemcpy(m_devData, data, m_sizeX * m_sizeY * sizeof(float), cudaMemcpyHostToDevice));
        }
        else
        {
            m_devData = NULL;
        }
    }
    else if (dataType == DeviceToHost)
    {
        m_devData = data;
        m_sizeX = sizeX;
        m_sizeY = sizeY;
    }
    else
    {
        printf("Wrong DataType\n");
    }
}

Tensor::~Tensor()
{
    cudaFree(m_devData);
}

int Tensor::getSize(Axis ax) const
{
    if (ax == X)
        return m_sizeX;
    else if (ax == Y)
        return m_sizeY;
    return -1;
}

float* Tensor::getDeviceData()
{
    return m_devData;
}

void Tensor::fetchDeviceData(float** hostData)
{
    *hostData = (float*)malloc(m_sizeX * m_sizeY * sizeof(float));
    gpuErrCheck(cudaMemcpy(*hostData,
                           m_devData,
                           m_sizeX * m_sizeY * sizeof(float),
                           cudaMemcpyDeviceToHost));       
}

/* KERNELS */ 

__global__ void addKernel(float* a, float* b, int sizeX, int sizeY)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < sizeX && y_idx < sizeY)
    {
        a[y_idx * sizeX + x_idx] += b[y_idx * sizeX + x_idx];
    }
}

__global__ void subtractKernel(float* a, float* b, int sizeX, int sizeY)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < sizeX && y_idx < sizeY)
    {
        a[y_idx * sizeX + x_idx] -= b[y_idx * sizeX + x_idx];
    }
}

void Tensor::add(Tensor& tensor)
{
    if (m_sizeX != tensor.getSize(X) || m_sizeY != tensor.getSize(Y))
    {
        printf("Tensors have to have the same shapes.\nTensor1: [%d, %d]\nTensor2: [%d, %d]\n",
               m_sizeX, m_sizeY, tensor.getSize(X), tensor.getSize(Y));
        exit(1);
    }

    dim3 blockSize(Config::addBlockSize, Config::addBlockSize, 1);
    dim3 gridSize((m_sizeX + blockSize.x - 1) / blockSize.x, (m_sizeY + blockSize.y - 1) / blockSize.y);
    addKernel<<<gridSize, blockSize>>>(getDeviceData(), tensor.getDeviceData(), m_sizeX, m_sizeY);
}

void Tensor::subtract(Tensor& tensor)
{
    if (m_sizeX != tensor.getSize(X) || m_sizeY != tensor.getSize(Y))
    {
        printf("Tensors have to have the same shapes.\nTensor1: [%d, %d]\nTensor2: [%d, %d]\n",
               m_sizeX, m_sizeY, tensor.getSize(X), tensor.getSize(Y));
        exit(1);
    }

    dim3 blockSize(Config::subtractBlockSize, Config::subtractBlockSize, 1);
    dim3 gridSize((m_sizeX + blockSize.x - 1) / blockSize.x, (m_sizeY + blockSize.y - 1) / blockSize.y);
    subtractKernel<<<gridSize, blockSize>>>(getDeviceData(), tensor.getDeviceData(), m_sizeX, m_sizeY);
}