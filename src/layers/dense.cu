#include "dense.cuh"

Dense::Dense(int input_size, int output_size)
{
    m_input_size = input_size;
    m_output_size = output_size;

    float* initWeights = new float[m_input_size * m_output_size];
    float* initBias = new float[m_output_size];

    for (int row = 0; row < m_output_size; row++)
    {
        initBias[row] = 0.0f;
        for (int col = 0; col < m_input_size; col++)
        {
            initWeights[row * m_input_size + col] = xavierInit(0, 1, m_input_size, m_output_size);
        }
    }

    m_weights = new Tensor(initWeights, m_output_size, m_input_size, HostToDevice);
    m_bias = new Tensor(initBias, m_output_size);
    m_delta_weights = NULL;
    m_delta_bias = NULL;
    m_z = NULL;
    m_gradients = NULL;

    delete[] initWeights;
    delete[] initBias;
}

Tensor* Dense::forward(Tensor* data)
{
    m_in_data = data;
    if (m_z == NULL)
    {
        m_z = new Tensor(m_weights->getSize(X),
                                    m_in_data->getSize(Y));
    }

    m_in_data->mul(m_weights, m_z);
    m_z->add(m_bias);

    return m_z;
}

Tensor* Dense::backward(Tensor* gradients)
{
    if (m_delta_weights == NULL)
    {
        m_delta_weights = new Tensor(gradients->getSize(X),
                                        m_in_data->getSize(X));
    }
    if (m_delta_bias == NULL)
    {
        m_delta_bias = new Tensor(gradients->getSize(X));
    }
    m_in_data->transposeMul(gradients, m_delta_weights);
    gradients->sumX(m_delta_bias);

    if (m_gradients == NULL)
    {
        m_gradients = new Tensor(m_weights->getSize(Y), gradients->getSize(Y));
    }
    gradients->mulTranspose(m_weights, m_gradients);
    return m_gradients;
}
