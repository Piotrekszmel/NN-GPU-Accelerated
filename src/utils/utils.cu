#include "utils.cuh"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
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

float xavierInit(int mean, int stddev, int num_input, int num_output)
{
   std::random_device rd;
   std::mt19937 e2(rd());
   std::normal_distribution<> dist(mean, stddev);
   float rand_num = dist(e2) * sqrt(2.0f / (num_input + num_output));
   return rand_num;
}

