#include "cross_entropy.cuh"
#include <iostream>

__global__ void crossEntropySoftMaxBackwardKernel(float *y_pred,
                                                  int size_x,
                                                  int size_y,
                                                  float* y_true,
                                                  float* gradients)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y)
    {
        float sum = 0.0f;

        for (int col = 0; col < size_x; col++)
        {
            sum += exp(y_pred[row * size_x + col]);
        }

        for (int col = 0; col < size_x; col++)
        {
            gradients[row * size_x + col] = (exp(y_pred[row * size_x + col]) / sum)
                                            - y_true[row * size_x + col];
        }
    }
}

__global__ void crossEntropySoftMaxLossKernel(float *y_pred,
                                              int size_x,
                                              int size_y,
                                              float* y_true,
                                              float* error)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size_y)
    {
        float sum = 0.0f;

        for (int col = 0; col < size_x; col++)
        {
            sum += exp(y_pred[row * size_x + col]);
        }
        float err = 0.0f;
        for (int col = 0; col < size_x; col++)
        {
            float val = exp(y_pred[row * size_x + col]) / sum;
            printf("val %f   ", val);
            err -= y_true[row * size_x + col] * log(val)
                   + (1 - y_true[row * size_x + col]) * log(1 - val); 
        }
        atomicAdd(error, err);
    }
}

__global__ void crossEntropySoftMaxAccuracyKernel(float *y_pred,
                                                 int size_x,
                                                 int size_y,
                                                 float* y_true,
                                                 float* accuracy)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size_y)
    {
        int max_idx = 0;
        int max_val = y_pred[row * size_x];

        for (int col = 1; col < size_x; col++)
        {
            if (y_pred[row * size_x + col] > max_val)
            {
                max_idx = col;
                max_val = y_pred[row * size_x + col];
            }
        }

        if (y_true[max_idx] == 1)
        {
            atomicAdd(accuracy, 1); // Revise it
        }
    }
}

float CrossEntropy::loss(Tensor* y_pred, Tensor* y_true)
{
    float error = 0.0f;
    float* d_error;

    cudaMalloc((void**)&d_error, sizeof(float));
    cudaMemcpy(d_error, &error, sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(Config::crossEntropyMetricBlockSize, 1, 1);
    dim3 gridSize((y_pred->getSize(Y) + blockSize.x - 1) / blockSize.x, 1, 1);

    crossEntropySoftMaxLossKernel<<<gridSize, blockSize>>>(y_pred->getDeviceData(),
                                  y_pred->getSize(X),
                                  y_pred->getSize(Y),
                                  y_true->getDeviceData(),
                                  d_error);
    cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_error);
    return error / y_pred->getSize(X);
}

float CrossEntropy::accuracy(Tensor* y_pred, Tensor* y_true)
{
    float accuracy = 0.0f;
    float* d_accuracy;

    cudaMalloc((void**)&d_accuracy, sizeof(float));
    cudaMemcpy(d_accuracy, &accuracy, sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(Config::crossEntropyMetricBlockSize, 1, 1);
    dim3 gridSize((y_pred->getSize(Y) + blockSize.x - 1) / blockSize.x, 1, 1);

    crossEntropySoftMaxAccuracyKernel<<<gridSize, blockSize>>>(y_pred->getDeviceData(),
                                                               y_pred->getSize(X),
                                                               y_pred->getSize(Y),
                                                               y_true->getDeviceData(),
                                                               d_accuracy);
    cudaMemcpy(&accuracy, d_accuracy, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_accuracy);
    return 100.0 * accuracy / y_pred->getSize(Y);
}

Tensor* CrossEntropy::backward(Tensor* y_pred, Tensor* y_true, Tensor* gradients)
{
    dim3 blockSize(Config::crossEntropyBlockSize, 1, 1);
    dim3 gridSize((y_pred->getSize(Y) + blockSize.x - 1) / blockSize.x, 1, 1);

    crossEntropySoftMaxBackwardKernel<<<gridSize, blockSize>>>(y_pred->getDeviceData(),
                                                               y_pred->getSize(X),
                                                               y_pred->getSize(Y),
                                                               y_true->getDeviceData(),
                                                               gradients->getDeviceData());
    return gradients;
}
