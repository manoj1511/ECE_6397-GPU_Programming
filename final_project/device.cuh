#ifndef DEVICE_CUH
#define DEVICE_CUH

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cmath>
#include <iostream>

void partial_derv(float *dx, float *dy, float *dz, int *pixel, int h, int h2, int width, int height, int depth);
void tensor(matrix *T_gpu, float *dx, float *dy, float *dz, int size);	
#endif
