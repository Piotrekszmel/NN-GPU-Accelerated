#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

class Config
{
private:
    static float strToFloat(std::string var, float defaultValue);
    static int strToInt(std::string var, int defaultValue);

public:
    /* CUDA CONFIGURATION */
    static int addBlockSize;
    static int subtractBlockSize;
    static int scaleBlockSize;
    static int multiplyBlockSize;
    static int meanBlockSize;
    static int sharedMemory;
    static int numBlocks;

    /* NEURAL NETWORK CONFIGURATION */
    static int epochs;
    static int batchSize;
    static float learningRate;
    static int reluBlockSize;    
    static int crossEntropyMetricBlockSize;
    static int crossEntropyBlockSize;

    static void printConfig();
    static void printCudaConfig();
    static void printNetConfig();
};
