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

    this->m_weights = new Tensor(initWeights, m_output_size, m_input_size, HostToDevice);
    this->m_bias = new Tensor(initBias, m_output_size);
    this->m_delta_weights = NULL;
    this->m_delta_bias = NULL;
    this->out_forward = NULL;
    this->out_backward = NULL;

    delete[] initWeights;
    delete[] initBias;
}

Tensor* Dense::forward(Tensor* data)
{
    this->in_data = data;
    if (this->out_forward == NULL)
    {
        this->out_forward = new Tensor(this->m_weights->getSize(X),
                                       this->in_data->getSize(Y));
    }

    this->in_data->mul(this->m_weights, this->out_forward);
    this->out_forward->add(this->m_bias);

    return this->out_forward;
}
