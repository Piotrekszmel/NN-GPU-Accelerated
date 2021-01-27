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
    Tensor(int sizeX, int sizeY = 1);
    Tensor(float* data, int sizeX, int sizeY = 1, DataType dataType = HostToDevice);
    ~Tensor();

    int getSize(Axis ax) const;
    float* getDeviceData(); 
    void fetchDeviceData(float** ptr);

    void add(Tensor& tensor);
    void subtract(Tensor& tensor);
    void scale(const float factor);
    void mul(Tensor& tensor, Tensor& output);
    void mulTransposition(Tensor& tensor, Tensor& output);
    void meanX(Tensor& output);

    void debug();

private: 
    int m_sizeX;
    int m_sizeY;
    float* m_devData;
};