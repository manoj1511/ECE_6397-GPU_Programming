#ifndef DEVICE_CUH
#define DEVICE_CUH

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <cmath>
#include <iostream>

struct matrix
{
	float a1, a2, a3;
	float b1, b2, b3;
	float c1, c2, c3;
};

void partial_derv(float *dx, float *dy, float *dz, int *pixel, int h, int h2, int width, int height, int depth);
void tensor(matrix *T_gpu, float *dx, float *dy, float *dz, int size);	

#endif
