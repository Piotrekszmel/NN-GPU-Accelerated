#pragma once

#include <cstdio>
#include <random>

#define gpuErrCheck(ans) {gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

#define DEBUG 0

#if defined(DEBUG) && DEBUG >= 1
    #define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
    #define DEBUG_PRINT(fmt, args...)
#endif

float randFloat(float min, float max);
float xavierInit(int mean, int stddev, int num_input, int num_output);
int randInt(int min, int max);