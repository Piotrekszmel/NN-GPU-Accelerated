#include "utils.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

float randFloat(float min, float max)
{
   return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (min - max)));
}

int randInt(int min, int max)
{
   return min + (rand() % static_cast<int>(max - min + 1));
}
