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


    Tensor tensor1(h_data, 2, 5, HostToDevice);
    Tensor tensor2(h_data2, 2, 5, HostToDevice);

    float* fetch_data = new float[10];

    tensor1.add(tensor2);
    tensor1.fetchDeviceData(&fetch_data);

    std::cout << "\n\n";
    for (int i = 0; i < 10; i++)
    {
        std::cout << h_data[i] << " + " << h_data2[i] << " = " << fetch_data[i] << "\n";
    }
    std::cout << std::endl;


}