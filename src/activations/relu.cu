#include "relu.cuh"

__global__ void reluKernel(float* A, int a_size_x, int a_size_y, float* output)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < a_size_x && y_idx < a_size_y)
    {
        if (A[y_idx * a_size_x + x_idx] >= 0.0f)
        {
            output[y_idx * a_size_x + x_idx] = A[y_idx * a_size_x + x_idx]; 
        }
        else
        {
            output[y_idx * a_size_x + x_idx] = 0.0f;
        }
    }
}

__global__ void reluBackpropKernel(float* A, int a_size_x, int a_size_y, float* output)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < a_size_x && y_idx < a_size_y)
    {
        if (A[y_idx * a_size_x + x_idx] >= 0.0f)
        {
            output[y_idx * a_size_x + x_idx] = 1.0f; 
        }
        else
        {
            output[y_idx * a_size_x + x_idx] = 0.0f;
        }
    }
}

ReLU::ReLU()
{
    m_a = NULL;
    m_gradients = NULL;
}

Tensor* ReLU::activation(Tensor* data)
{
    if (m_a == NULL)
    {
        m_a = new Tensor(data->getSize(X), data->getSize(Y));
    }

    dim3 blockSize(Config::reluBlockSize, Config::reluBlockSize, 1);
    dim3 gridSize((data->getSize(X) + blockSize.x - 1) / blockSize.x,
                  (data->getSize(Y) + blockSize.y - 1) / blockSize.y, 1);
    reluKernel<<<gridSize, blockSize>>>(data->getDeviceData(),
                                        data->getSize(X), 
                                        data->getSize(Y),
                                        m_a->getDeviceData());
    return m_a;
}

Tensor* ReLU::derivative(Tensor* gradients)
{
    if (m_gradients == NULL)
    {
        m_gradients = new Tensor(gradients->getSize(X), gradients->getSize(Y));
    }
    dim3 blockSize(Config::reluBlockSize, Config::reluBlockSize, 1);
    dim3 gridSize((gradients->getSize(X) + blockSize.x - 1) / blockSize.x,
                  (gradients->getSize(Y) + blockSize.y - 1) / blockSize.y, 1);
    reluBackpropKernel<<<gridSize, blockSize>>>(gradients->getDeviceData(),
                                                gradients->getSize(X),
                                                gradients->getSize(Y),
                                                m_gradients->getDeviceData());
    return m_gradients;
}
