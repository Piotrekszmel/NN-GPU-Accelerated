#include "dense.cuh"

Dense::Dense(int input_size, int output_size)
{
    m_input_size = input_size;
    m_output_size = output_size;

    float* initWeights = new float[m_input_size * m_output_size];

    for (int row = 0; row < m_output_size; row++)
    {
        for (int col = 0; col < m_input_size; col++)
        {
            initWeights[row * m_input_size + col] = randFloat(0, 1);
        }
    }

    m_weights = Tensor(initWeights, m_output_size, m_input_size, HostToDevice);
}