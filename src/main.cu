#include "Tensor/tensor.cuh"
#include <iostream>

int main()
{
    
    float* h_data = new float[10];
    for (int i = 0; i < 10; i++)
        h_data[i] = i;
    
    float* h_data2 = new float[10];
    for (int i = 0; i < 10; i++)
        h_data2[i] = i + i;


    Tensor tensor1(h_data, 2, 5, HOST);
    Tensor tensor2(h_data2, 2, 5, HOST);

    float* fetch_data = NULL;

    tensor1.fetchDeviceData(&fetch_data);

    for (int i = 0; i < 10; i++)
        std::cout << fetch_data[i] << "\n";
    std::cout << std::endl;


}