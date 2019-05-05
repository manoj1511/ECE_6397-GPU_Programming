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

struct eigen_vectors
{
	float x, y, z;
};

void partial_derv(float *dx, float *dy, float *dz, int *pixel, int h, int h2, int width, int height, int depth);
void tensor(matrix *T, float *dx, float *dy, float *dz, int size);	
void filter_x(matrix *T, matrix *T_x, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth);
void filter_y(matrix *T_x, matrix *T_y, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth);
void filter_z(matrix *T_y, matrix *T_z, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth);
void remove_padding(matrix *T_z, matrix *T_m, int half, int half_z, int width, int height, int depth);
#endif
