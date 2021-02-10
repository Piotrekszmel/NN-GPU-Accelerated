#include "config.cuh"

int Config::addBlockSize = 8;
int Config::subtractBlockSize = 8;
int Config::scaleBlockSize = 8;
int Config::multiplyBlockSize = 8;
int Config::meanBlockSize = 8;
int Config::sumBlockSize = 8;
int Config::sharedMemory = 0;
int Config::numBlocks = -1;

int Config::epochs = 100;
int Config::batchSize = 512;
float Config::learningRate = 1e-02;
int Config::reluBlockSize = 8;
int Config::crossEntropyMetricBlockSize = 64;
int Config::crossEntropyBlockSize = 64;

float Config::strToFloat(std::string var, float defaultValue)
{
    char* value = std::getenv(var.c_str());
    return value == NULL ? defaultValue : atof(value);
}

int Config::strToInt(std::string var, int defaultValue)
{
    char* value = std::getenv(var.c_str());
    return value == NULL ? defaultValue : atoi(value);
}

void Config::printConfig()
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
    printf("=====================================\n");
    printf("            Configuration\n");
    printf("=====================================\n");
    printf("                CUDA\n");
    printf("=====================================\n");
    printf(" Device name: %s\n", properties.name);
    printf(" Memory Clock Rate (KHz): %d\n", properties.memoryClockRate);
    printf(" Memory Bus Width (bits): %d\n", properties.memoryBusWidth);
    printf("-------------------------------------\n");
    printf(" AddBlockSize: %d\n", Config::addBlockSize);
    printf(" SubtractBlockSize: %d\n", Config::subtractBlockSize);
    printf(" ScaleBlockSize: %d\n", Config::scaleBlockSize);
    printf(" MultiplyBlockSize: %d\n", Config::multiplyBlockSize);
    printf(" MeanBlockSize: %d\n", Config::meanBlockSize);
    printf(" SumBlockSize: %d\n", Config::sumBlockSize);
    printf(" SharedMemory: %d\n", Config::sharedMemory);
    printf(" numBlocks: %d\n", Config::numBlocks);
    printf("=====================================\n");
    printf("            Neural Network\n");
    printf("=====================================\n");
    printf(" Epochs: %d\n", Config::epochs);
    printf(" BatchSize: %d\n", Config::batchSize);
    printf(" LearningRate: %e\n", Config::learningRate);
    printf("\n");
    printf(" ReLuBlockSize: %d\n", Config::reluBlockSize);
    printf("\n");
    printf(" CrossEntropyMetricBlockSize: %d\n", Config::crossEntropyMetricBlockSize);
    printf(" CrossEntropyBlockSize: %d\n", Config::crossEntropyBlockSize);
    
    printf("=====================================\n");
    printf("\n");
}

void Config::printCudaConfig()
{
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    
    printf("=====================================\n");
    printf("            CUDA Configuration\n");
    printf("=====================================\n");
    printf(" Device name: %s\n", properties.name);
    printf(" Memory Clock Rate (KHz): %d\n", properties.memoryClockRate);
    printf(" Memory Bus Width (bits): %d\n", properties.memoryBusWidth);
    printf("-------------------------------------\n");
    printf(" AddBlockSize: %d\n", Config::addBlockSize);
    printf(" SubtractBlockSize: %d\n", Config::subtractBlockSize);
    printf(" ScaleBlockSize: %d\n", Config::scaleBlockSize);
    printf(" MultiplyBlockSize: %d\n", Config::multiplyBlockSize);
    printf(" MeanBlockSize: %d\n", Config::meanBlockSize);
    printf(" SumBlockSize: %d\n", Config::sumBlockSize);
    printf(" SharedMemory: %d\n", Config::sharedMemory);
    printf(" numBlocks: %d\n", Config::numBlocks);

    printf("=====================================\n");
    printf("\n");
}

void Config::printNetConfig()
{
    printf("=====================================\n");
    printf("            Neural Network Configuration\n");
    printf("=====================================\n");
    printf(" Epochs: %d\n", Config::epochs);
    printf(" BatchSize: %d\n", Config::batchSize);
    printf(" LearningRate: %e\n", Config::learningRate);
    printf("\n");
    printf(" ReLuBlockSize: %d\n", Config::reluBlockSize);
    printf("\n");
    printf(" CrossEntropyMetricBlockSize: %d\n", Config::crossEntropyMetricBlockSize);
    printf(" CrossEntropyBlockSize: %d\n", Config::crossEntropyBlockSize);
    
    printf("=====================================\n");
    printf("\n");
}
