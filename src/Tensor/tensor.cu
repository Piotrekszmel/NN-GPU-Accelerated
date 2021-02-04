#include "tensor.cuh"
#include "../config/config.cuh"

/* CONSTRUCTORS */ 
Tensor::Tensor(int size_x, int size_y)
{
    m_size_x = size_x;
    m_size_y = size_y;

    if (m_size_x && m_size_y)
    {
        cudaMalloc((void**)&m_devData, m_size_x * m_size_y * sizeof(float));
    }
    else
    {
        m_devData = NULL;
    }
}

Tensor::Tensor(float* data, int size_x, int size_y, DataType dataType)
{
    m_size_x = size_x;
    m_size_y = size_y;
    if (dataType == HostToDevice)
    {
        if (m_size_x && m_size_y)
        {
            gpuErrCheck(cudaMalloc((void**)&m_devData, m_size_x * m_size_y * sizeof(float)));
            gpuErrCheck(cudaMemcpy(m_devData, data, m_size_x * m_size_y * sizeof(float), cudaMemcpyHostToDevice));
        }
        else
        {
            m_devData = NULL;
        }
    }
    else if (dataType == DeviceToHost)
    {
        m_devData = data;
        m_size_x = size_x;
        m_size_y = size_y;
    }
    else
    {
        printf("Wrong DataType\n");
    }
}

Tensor::~Tensor()
{
    cudaFree(m_devData);
}

int Tensor::getSize(Axis ax) const
{
    if (ax == X)
        return m_size_x;
    else if (ax == Y)
        return m_size_y;
    return -1;
}

void Tensor::setSize(Axis ax, int size)
{
    if (ax == X)
        m_size_x = size;
    else if (ax == Y)
        m_size_y = size;
    else
    {
        printf("Wrong axis provided!\n");
        exit(1);
    }

}

float* Tensor::getDeviceData()
{
    return m_devData;
}

void Tensor::fetchDeviceData(float** hostData)
{
    *hostData = (float*)malloc(m_size_x * m_size_y * sizeof(float));
    gpuErrCheck(cudaMemcpy(*hostData,
                           m_devData,
                           m_size_x * m_size_y * sizeof(float),
                           cudaMemcpyDeviceToHost));       
}

/* KERNELS */ 
__global__ void addKernel(float* a, float* b, int size_x, int size_y)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < size_x && y_idx < size_y)
    {
        a[y_idx * size_x + x_idx] += b[y_idx * size_x + x_idx];
    }
}

__global__ void subtractKernel(float* a, float* b, int size_x, int size_y)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < size_x && y_idx < size_y)
    {
        a[y_idx * size_x + x_idx] -= b[y_idx * size_x + x_idx];
    }
}

__global__ void scaleKernel(float* a, float factor, int size_x, int size_y)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < size_x && y_idx < size_y)
    {
        a[y_idx * size_x + x_idx] *= factor;
    }
}

__global__ void multiplyKernel(float* A, float* B,
                          int a_size_x, int a_size_y,
                          int b_size_x, int b_size_y,
                          int fields_per_block_x, 
                          int fields_per_block_y,
                          int fields_per_thread_x,
                          int fields_per_thread_y,
                          float* C)
{
    int block_x_start = blockIdx.x * fields_per_block_x;
    int block_y_start = blockIdx.y * fields_per_block_y;
    int block_x_end = min(b_size_x, block_x_start + fields_per_block_x);
    int block_y_end = min(a_size_y, block_y_start + fields_per_block_y);
    int thread_x_start = threadIdx.x * fields_per_thread_x;
    int thread_y_start = threadIdx.y * fields_per_thread_y;
    int thread_x_end = thread_x_start + fields_per_thread_x;
    int thread_y_end = thread_y_start + fields_per_thread_y;

    int start_idx_x = block_x_start + thread_x_start;
    int start_idx_y = block_y_start + thread_y_start;
    int end_idx_x = min(block_x_end, block_x_start + thread_x_end);
    int end_idx_y = min(block_y_end, block_y_start + thread_y_end);

    for (int y = start_idx_y; y < end_idx_y; y++)
    {
        for (int x = start_idx_x; x < end_idx_x; x++)
        {
            float sum = 0.0f;
            for (int i = 0; i < a_size_x; i++)
            {
                sum += A[y * a_size_x + i] * B[i * b_size_x + x];
            }
            C[y * b_size_x + x] = sum;    
        }
    }
}

__global__ void multiplySharedMemoryKernel(float* A, float* B,
                                           int a_size_x, int a_size_y,
                                           int b_size_x, int b_size_y,
                                           float* C)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.x * blockDim.x + threadIdx.y;
    int chunks = (a_size_x + blockDim.x - 1) / blockDim.x;
    if (x_idx < b_size_x && y_idx < a_size_y)
    {
        extern __shared__ float array[];
        float* s_A = (float*)array;
        float* s_B = (float*)&s_A[blockDim.x * blockDim.y];

        float sum = 0.0f;
        for (int i = 0; i < chunks; i++)
        {
            if (i * blockDim.x + threadIdx.x < a_size_x && blockIdx.y * blockDim.y + threadIdx.y < a_size_y)
            {
                s_A[threadIdx.y * blockDim.x + threadIdx.x] 
                    = A[(blockIdx.y * blockDim.y + threadIdx.y) * a_size_x + i *  blockDim.x + threadIdx.x]; 
            }
            else
            {
                s_A[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }
            if (blockIdx.x * blockDim.x + threadIdx.x < b_size_x && i * blockDim.y + threadIdx.y < b_size_y)
            {
                s_B[threadIdx.y * blockDim.x + threadIdx.x] 
                    = B[(i * blockDim.y + threadIdx.y) * b_size_x + blockIdx.x * blockDim.x + threadIdx.x];
            }
            else
            {
                s_B[threadIdx.y * blockDim.x + threadIdx.x]  = 0.0f;
            }

            __syncthreads();
            for (int i = 0; i < blockDim.x; i++)
                sum += s_A[threadIdx.y * blockDim.x + i] * s_B[i * blockDim.x + threadIdx.x]; 
            __syncthreads();
        }
        C[y_idx * b_size_x + x_idx] = sum;
    }

}

__global__ void multiplyByTranposeKernel(float* A, float* B,
                                         int a_size_x, int a_size_y,
                                         int b_size_x, int b_size_y,
                                         int fields_per_block_x, 
                                         int fields_per_block_y,
                                         int fields_per_thread_x,
                                         int fields_per_thread_y,
                                         float* C)
{
    int block_x_start = blockIdx.x * fields_per_block_x;
    int block_y_start = blockIdx.y * fields_per_block_y;
    int block_x_end = min(b_size_y, block_x_start + fields_per_block_x);
    int block_y_end = min(a_size_y, block_y_start + fields_per_block_y);
    int thread_x_start = threadIdx.x * fields_per_thread_x;
    int thread_y_start = threadIdx.y * fields_per_thread_y;
    int thread_x_end = thread_x_start + fields_per_thread_x;
    int thread_y_end = thread_y_start + fields_per_thread_y;
    int start_idx_x = block_x_start + thread_x_start;
    int start_idx_y = block_y_start + thread_y_start;
    int end_idx_x = min(block_x_end, block_x_start + thread_x_end);
    int end_idx_y = min(block_y_end, block_y_start + thread_y_end);

    for (int y = start_idx_y; y < end_idx_y; y++)
    {
        for (int x = start_idx_x; x < end_idx_x; x++)
        {
            float sum = 0.0;
            for (int i = 0; i < a_size_x; i++)
            {
                sum += A[y * a_size_x + i] * B[x * b_size_x + i];
            }
            C[y * b_size_y + x] = sum;
        }
    }
}

__global__ void multiplyByTranposeSharedMemoryKernel(float* A, float* B,
                                                     int a_size_x, int a_size_y,
                                                     int b_size_x, int b_size_y,
                                                     float* C)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int chunks = (a_size_x + blockDim.x - 1) / blockDim.x;
    
    if (x_idx < b_size_y && y_idx < a_size_y)
    {
        extern __shared__ float array[];
        float* s_A = (float*)array;
        float* s_B = (float*)&array[blockDim.x * blockDim.y];
        float sum = 0.0f;

        for (int i = 0; i < chunks; i++)
        {
            if (i * blockDim.x + threadIdx.x < a_size_x 
                && blockIdx.y * blockDim.y + threadIdx.y < a_size_y)
            {
                s_A[threadIdx.y * blockDim.x + threadIdx.x] 
                    = A[(blockIdx.y * blockDim.y + threadIdx.y) * a_size_x + i * blockDim.x + threadIdx.x];
            }
            else
            {
                s_A[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }
            if (i * blockDim.x + threadIdx.x < b_size_x 
                && blockIdx.x * blockDim.y + threadIdx.y < b_size_y)
            {
                s_B[threadIdx.y * blockDim.x + threadIdx.x] 
                    = B[(blockIdx.x * blockDim.y + threadIdx.y) * b_size_x + i * blockDim.x + threadIdx.x];
            }
            else
            {
                s_B[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }

            __syncthreads();
            for (int j = 0; j < blockDim.x; j++)
            {
                sum += s_A[threadIdx.y * blockDim.x + j] * s_B[threadIdx.x * blockDim.x + j];
            }
            __syncthreads();
        }
        C[y_idx * b_size_y + x_idx] = sum;
    }
}

__global__ void transposeMultiplyKernel(float* A, float* B,
    int a_size_x, int a_size_y,
    int b_size_x, int b_size_y,
    int fields_per_block_x, 
    int fields_per_block_y,
    int fields_per_thread_x,
    int fields_per_thread_y,
    float* C)
{
    int block_x_start = blockIdx.x * fields_per_block_x;
    int block_y_start = blockIdx.y * fields_per_block_y;
    int block_x_end = min(b_size_x, block_x_start + fields_per_block_x);
    int block_y_end = min(a_size_x, block_y_start + fields_per_block_y);
    int thread_x_start = threadIdx.x * fields_per_thread_x;
    int thread_y_start = threadIdx.y * fields_per_thread_y;
    int thread_x_end = thread_x_start + fields_per_thread_x;
    int thread_y_end = thread_y_start + fields_per_thread_y;
    int start_idx_x = block_x_start + thread_x_start;
    int start_idx_y = block_y_start + thread_y_start;
    int end_idx_x = min(block_x_end, block_x_start + thread_x_end);
    int end_idx_y = min(block_y_end, block_y_start + thread_y_end);

    for (int y = start_idx_y; y < end_idx_y; y++)
    {
        for (int x = start_idx_x; x < end_idx_x; x++)
        {
            float sum = 0.0;
            for (int i = 0; i < b_size_y; i++)
            {
                sum += A[i * a_size_x + y] * B[i * b_size_x + x];
            }
            C[y * b_size_x + x] = sum;
        }
    }
}

__global__ void tranposeMultiplySharedMemoryKernel(float* A, float* B,
    int a_size_x, int a_size_y,
    int b_size_x, int b_size_y,
    float* C)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int chunks = (a_size_y + blockDim.y - 1) / blockDim.y;

    if (x_idx < b_size_x && y_idx < a_size_x)
    {
        extern __shared__ float array[];
        float* s_A = (float*)array;
        float* s_B = (float*)&array[blockDim.x * blockDim.y];
        float sum = 0.0f;
        
        for (int i = 0; i < chunks; i++)
        {
            if (i * blockDim.y + threadIdx.y < a_size_y 
                && blockIdx.y * blockDim.x + threadIdx.x < a_size_x)
            {
                s_A[threadIdx.y * blockDim.x + threadIdx.x] 
                    = A[(i * blockDim.y + threadIdx.y) * a_size_x + blockIdx.y * blockDim.x + threadIdx.x];
            }
            else
            {
                s_A[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }
            if (i * blockDim.x + threadIdx.y < b_size_y 
                && blockIdx.x * blockDim.y + threadIdx.x < b_size_x)
            {
                s_B[threadIdx.y * blockDim.x + threadIdx.x] 
                    = B[(i * blockDim.y + threadIdx.y) * b_size_x + blockIdx.x * blockDim.x + threadIdx.x];
            }
            else
            {
                s_B[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
            }

            __syncthreads();
            for (int j = 0; j < blockDim.x; j++)
            {
                sum += s_A[j * blockDim.x + threadIdx.y] * s_B[j * blockDim.x + threadIdx.x];
            }
            __syncthreads();
        }
        
        C[y_idx * b_size_x + x_idx] = sum;
    }
}

__global__ void meanXKernel(float* A, int size_x, int size_y, float* B)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < size_x)
    {
        float sum = 0.0;
        for (int i = 0; i < size_y; i++)
        {
            sum += A[i * size_x + col];
        }

        B[col] = sum / size_y;
    }
}

/* OPERATIONS */
void Tensor::add(Tensor& tensor)
{
    if (m_size_x != tensor.getSize(X) || m_size_y != tensor.getSize(Y))
    {
        printf("Tensors have to have the same shapes.\nTensor1: [%d, %d]\nTensor2: [%d, %d]\n",
               m_size_x, m_size_y, tensor.getSize(X), tensor.getSize(Y));
        exit(1);
    }

    dim3 blockSize(Config::addBlockSize, Config::addBlockSize, 1);
    dim3 gridSize((m_size_x + blockSize.x - 1) / blockSize.x, (m_size_y + blockSize.y - 1) / blockSize.y, 1);
    addKernel<<<gridSize, blockSize>>>(getDeviceData(), tensor.getDeviceData(), m_size_x, m_size_y);
}

void Tensor::subtract(Tensor& tensor)
{
    if (m_size_x != tensor.getSize(X) || m_size_y != tensor.getSize(Y))
    {
        printf("Tensors have to have the same shapes.\nTensor1: [%d, %d]\nTensor2: [%d, %d]\n",
               m_size_x, m_size_y, tensor.getSize(X), tensor.getSize(Y));
        exit(1);
    }

    dim3 blockSize(Config::subtractBlockSize, Config::subtractBlockSize, 1);
    dim3 gridSize((m_size_x + blockSize.x - 1) / blockSize.x, (m_size_y + blockSize.y - 1) / blockSize.y, 1);
    subtractKernel<<<gridSize, blockSize>>>(getDeviceData(), tensor.getDeviceData(), m_size_x, m_size_y);
}

void Tensor::scale(float factor)
{
    dim3 blockSize(Config::scaleBlockSize, Config::scaleBlockSize, 1);
    dim3 gridSize((m_size_x + blockSize.x - 1) / blockSize.x, (m_size_y + blockSize.y - 1) / blockSize.y, 1);
    scaleKernel<<<gridSize, blockSize>>>(getDeviceData(), factor, m_size_x, m_size_y);
}

void Tensor::mul(Tensor& tensor, Tensor& output)
{
    if (m_size_x != tensor.getSize(Y))
    {
        printf("first dim of the first tensor has to be equal to the second dim of the second tensor. Got: "
              "[%d, %d], [%d, %d]\n", m_size_x, m_size_y, tensor.getSize(X), tensor.getSize(Y));
        exit(1);
    }

    if (Config::sharedMemory == 1)
    {
        dim3 blockSize(Config::multiplyBlockSize, Config::multiplyBlockSize, 1);
        dim3 gridSize((m_size_x + blockSize.x - 1) / blockSize.x, (m_size_y + blockSize.y - 1) / blockSize.y, 1);
        int sharedMemory = 2 * blockSize.x * blockSize.y * sizeof(float);
        multiplySharedMemoryKernel<<<gridSize, blockSize, sharedMemory>>>(getDeviceData(),
                                   tensor.getDeviceData(),
                                  m_size_x,
                                  m_size_y,
                                   tensor.getSize(X),
                                   tensor.getSize(Y),
                                   output.getDeviceData());
    }
    else
    {
        int num_threads = Config::multiplyBlockSize;
        int num_blocks_x = Config::numBlocks == -1
                            ? (tensor.getSize(X) + num_threads - 1) / num_threads
                            : Config::numBlocks;
        int num_blocks_y = Config::numBlocks == -1
        ? (m_size_y + num_threads - 1) / num_threads
        : Config::numBlocks;   
        int fields_per_block_x = max(1, (tensor.getSize(X) + num_blocks_x - 1) / num_blocks_x);
        int fields_per_block_y = max(1, (getSize(Y) + num_blocks_y - 1) / num_blocks_y);
        int fields_per_thread_x = max(1, (fields_per_block_x + num_threads - 1) / num_threads);
        int fields_per_thread_y = max(1, (fields_per_block_y + num_threads - 1) / num_threads);

        dim3 gridSize(num_blocks_x, num_blocks_y, 1);
        dim3 blockSize(num_threads, num_threads, 1);
        multiplyKernel<<<gridSize, blockSize>>>(getDeviceData(), 
                       tensor.getDeviceData(),
                      m_size_x, 
                      m_size_y,
                       tensor.getSize(X), 
                       tensor.getSize(Y),
                       fields_per_block_x,
                       fields_per_block_y,
                       fields_per_thread_x,
                       fields_per_thread_y,
                       output.getDeviceData());
    }
}

void Tensor::mulTranspose(Tensor& tensor, Tensor& output)
{
    if (m_size_x != tensor.getSize(X))
    {
        printf("first dim of the first tensor has to be equal to the first dim of the second tensor. Got: "
              "[%d, %d], [%d, %d]\n", m_size_x, m_size_y, tensor.getSize(X), tensor.getSize(Y));
        exit(1);
    }
    if(Config::sharedMemory == 1 && m_size_x > m_size_y && tensor.getSize(X) > tensor.getSize(Y))
    {
        printf("mulTranspose does not support shared memory if first dim is greater than second dim "
               "Got: Tensor1: [%d, %d],  Tensor2: [%d, %d].\nGlobal memory will be used!\n",
                m_size_x, m_size_y, tensor.getSize(X), tensor.getSize(Y));
    }

    if (Config::sharedMemory == 1 && m_size_x <= m_size_y && tensor.getSize(X) <= tensor.getSize(Y))
    {
        dim3 blockSize(Config::multiplyBlockSize, Config::multiplyBlockSize, 1);
        dim3 gridSize((tensor.getSize(Y) + blockSize.x - 1) / blockSize.x, (m_size_y + blockSize.y - 1) / blockSize.y, 1);
        int sharedMemory = 2 * blockSize.x * blockSize.y * sizeof(float);

        multiplyByTranposeSharedMemoryKernel<<<gridSize, blockSize, sharedMemory>>>(getDeviceData(),
                                   tensor.getDeviceData(),
                                  m_size_x,
                                  m_size_y,
                                   tensor.getSize(X),
                                   tensor.getSize(Y),
                                   output.getDeviceData());
    }
    else
    {
        int num_threads = Config::multiplyBlockSize;
        int num_blocks_x = Config::numBlocks == -1
                            ? (tensor.getSize(Y) + num_threads - 1) / num_threads
                            : Config::numBlocks;
        int num_blocks_y = Config::numBlocks == -1
        ? (m_size_y + num_threads - 1) / num_threads
        : Config::numBlocks;   
        int fields_per_block_x = max(1, (tensor.getSize(Y) + num_blocks_x - 1) / num_blocks_x);
        int fields_per_block_y = max(1, (m_size_y + num_blocks_y - 1) / num_blocks_y);
        int fields_per_thread_x = max(1, (fields_per_block_x + num_threads - 1) / num_threads);
        int fields_per_thread_y = max(1, (fields_per_block_y + num_threads - 1) / num_threads);

        dim3 gridSize(num_blocks_x, num_blocks_y, 1);
        dim3 blockSize(num_threads, num_threads, 1);
        multiplyByTranposeKernel<<<gridSize, blockSize>>>(getDeviceData(), 
                       tensor.getDeviceData(),
                       m_size_x, 
                       m_size_y,
                       tensor.getSize(X), 
                       tensor.getSize(Y),
                       fields_per_block_x,
                       fields_per_block_y,
                       fields_per_thread_x,
                       fields_per_thread_y,
                       output.getDeviceData());
    }
}

void Tensor::transposeMul(Tensor& tensor, Tensor& output)
{
    if (m_size_y != tensor.getSize(Y))
    {
        printf("second dim of the first tensor has to be equal to the second dim of the second tensor. Got: "
              "[%d, %d], [%d, %d]\n", m_size_x, m_size_y, tensor.getSize(X), tensor.getSize(Y));
        exit(1);
    }

    if(Config::sharedMemory == 1 && m_size_x < m_size_y && tensor.getSize(X) < tensor.getSize(Y))
    {
        printf("transposeMul does not support shared memory if second dim is greater than first dim "
               "Got: Tensor1: [%d, %d],  Tensor2: [%d, %d].\nGlobal memory will be used!\n",
                m_size_x, m_size_y, tensor.getSize(X), tensor.getSize(Y));
    }

    if (Config::sharedMemory == 1 && m_size_x >= m_size_y && tensor.getSize(X) >= tensor.getSize(Y))
    {
        dim3 blockSize(Config::multiplyBlockSize, Config::multiplyBlockSize, 1);
        dim3 gridSize((tensor.getSize(X) + blockSize.x - 1) / blockSize.x, (m_size_x + blockSize.y - 1) / blockSize.y, 1);
        int sharedMemory = 2 * blockSize.x * blockSize.y * sizeof(float);

       tranposeMultiplySharedMemoryKernel<<<gridSize, blockSize, sharedMemory>>>(getDeviceData(),
                                   tensor.getDeviceData(),
                                   m_size_x,
                                   m_size_y,
                                   tensor.getSize(X),
                                   tensor.getSize(Y),
                                   output.getDeviceData());
    }
    else
    {
        int num_threads = Config::multiplyBlockSize;
        int num_blocks_x = Config::numBlocks == -1
                            ? (tensor.getSize(X) + num_threads - 1) / num_threads
                            : Config::numBlocks;
        int num_blocks_y = Config::numBlocks == -1
        ? (m_size_x + num_threads - 1) / num_threads
        : Config::numBlocks;   
        int fields_per_block_x = max(1, (tensor.getSize(X) + num_blocks_x - 1) / num_blocks_x);
        int fields_per_block_y = max(1, (getSize(X) + num_blocks_y - 1) / num_blocks_y);
        int fields_per_thread_x = max(1, (fields_per_block_x + num_threads - 1) / num_threads);
        int fields_per_thread_y = max(1, (fields_per_block_y + num_threads - 1) / num_threads);

        dim3 gridSize(num_blocks_x, num_blocks_y, 1);
        dim3 blockSize(num_threads, num_threads, 1);
        transposeMultiplyKernel<<<gridSize, blockSize>>>(getDeviceData(), 
                       tensor.getDeviceData(),
                       m_size_x, 
                       m_size_y,
                       tensor.getSize(X), 
                       tensor.getSize(Y),
                       fields_per_block_x,
                       fields_per_block_y,
                       fields_per_thread_x,
                       fields_per_thread_y,
                       output.getDeviceData());
    }
}

void Tensor::meanX(Tensor& output)
{
    int blockSize = Config::meanBlockSize;
    int gridSize = (m_size_x + blockSize - 1) / blockSize;
    meanXKernel<<<gridSize, blockSize>>>(getDeviceData(),
                                         m_size_x,
                                         m_size_y,
                                         output.getDeviceData());
}
