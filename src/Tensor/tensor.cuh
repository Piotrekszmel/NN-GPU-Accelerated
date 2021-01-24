#pragma once

#include <cstdio>

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
    Tensor(float* hostData, int sizeX, int sizeY = 1);
    Tensor(float* devData, int sizeX, int sizeY = 1);
    ~Tensor();

    int getSize(const Axis ax);
    float* getDeviceData();
    float** fetchDeviceData();

    void add(const Tensor& tensor);
    void subtract(const Tensor& tensor);
    void scale(const float factor);
    void mul(const Tensor& tensor, Tensor& output);
    void mulTransposition(const Tensor& tensor, Tensor& output);
    void meanX(Tensor& output);

    void debug();

private: 
    int m_sizeX;
    int m_sizeY;
    float* m_devData;
}