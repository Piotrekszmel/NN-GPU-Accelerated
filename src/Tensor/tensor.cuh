#pragma once

#include <cstdio>

#include "../utils/utils.cuh"

enum DataType 
{
    HostToDevice,
    DeviceToHost
};

enum Axis
{
    X, 
    Y
};

class Tensor
{
public:
    /* Constructors */
    Tensor(int size_x, int size_y = 1);
    Tensor(float* data, int size_x, int size_y = 1, DataType dataType = HostToDevice);
    ~Tensor();

    int getSize(Axis ax) const;
    float* getDeviceData(); 
    void fetchDeviceData(float** ptr);

    void add(Tensor& tensor);
    void subtract(Tensor& tensor);
    void scale(float factor);
    void mul(Tensor& tensor, Tensor& output);
    void mulByTranspose(Tensor& tensor, Tensor& output);
    void transposeMul(Tensor& tensor, Tensor& output);
    void meanX(Tensor& output);
    void multSharedMem(Tensor& tensor, Tensor& output);
    void debug();

private: 
    int m_size_x;
    int m_size_y;
    float* m_devData;
};