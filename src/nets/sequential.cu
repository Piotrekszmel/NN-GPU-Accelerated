#include "sequential.cuh"

Sequential::Sequential(Loss* loss_fn, Optimizer* optim)
{
    m_optim = optim;
    m_loss_fn = loss_fn;
    m_gradients = NULL;
}

void Sequential::addLayer(Layer* layer)
{
    m_layers.push_back(layer);
}

Tensor* Sequential::forward(Tensor* input)
{
    Tensor* X = input;
    std::vector<Layer*>::iterator layer;
    for (layer = m_layers.begin(); layer != m_layers.end(); layer++)
    {
        X = (*layer)->forward(X);
    }
    return X;
}

void Sequential::backward(Tensor* y_pred, Tensor* y_true)
{
    if (m_gradients == NULL)
    {
        m_gradients = new Tensor(y_pred->getSize(X), y_pred->getSize(Y));
    }

    m_loss_fn->backward(y_pred, y_true, m_gradients);

    Tensor* grads = m_gradients;
    std::vector<Layer*>::reverse_iterator rlayer;

    for (rlayer = m_layers.rbegin(); rlayer != m_layers.rend(); rlayer++)
    {
        grads = (*rlayer)->backward(grads);
    }

    std::vector<Layer*>::iterator layer;
    for (layer = m_layers.begin(); layer != m_layers.end(); layer++)
    {
        m_optim->step(*layer);
    }
}
