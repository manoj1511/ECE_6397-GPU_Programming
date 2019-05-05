#include "device.cuh"


__global__
void partial_derv_gpu(float *dx, float *dy, float *dz, int *pixel, int h, int h2, int width, int height, int depth)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	int index = (idz * width * height)+(idy * width)+idx;

//	extern __shared__ int Ptr[];
	
	if( (idx < width) && (idy < height) && (idz < depth) )
	{
		if(idx == 0) 			dx[index] = (pixel[index+1] - pixel[index])   / (float)h;
		else if(idx == width-1) 	dx[index] = (pixel[index]   - pixel[index-1]) / (float)h;
		else 				dx[index] = (pixel[index+1] - pixel[index-1]) / (2.0f*h);

		if(idy == 0) 			dy[index] = (pixel[width+index] - pixel[index])       / (float)h;
		else if(idy == height-1) 	dy[index] = (pixel[index] 	- pixel[index-width]) / (float)h;
		else 				dy[index] = (pixel[index+width] - pixel[index-width]) / (2.0f*h);

		if(idz == 0)			dz[index] = (pixel[index+width*height] - pixel[index]) 		    / (float)h2;
		else if(idz == depth - 1)	dz[index] = (pixel[index] 	       - pixel[index-width*height]) / (float)h2;
		else				dz[index] = (pixel[index+width*height] - pixel[index-width*height]) / (2.0f*h2);
	}
	else return;
}

__global__
void tensor_gpu(matrix *T, float *dx, float *dy, float *dz, int size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < size)
	{
		float x = dx[idx], y = dy[idx], z = dz[idx];
		
		T[idx].a1 = x * x;
		T[idx].a2 = x * y; 
		T[idx].a3 = x * z;
		T[idx].b2 = y * y;
		T[idx].b3 = y * z;
		T[idx].c3 = z * z;
	}
	else return;
}

__global__
void filter_x_gpu(matrix *T, matrix *T_x, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	int index = (idz * width * height)+(idy * width)+idx;


	if(idx <= (width - filter_size) && idy <= (height - half) && idz <= (depth - half_z))
	{

		float temp_a1, temp_a2, temp_a3, temp_b2, temp_b3, temp_c3;

              	int local_idx = idx + half;
            	int local_index = (idz * width * height) + (idy * width) + local_idx;

		for(int ref = 0; ref < filter_size; ref++)
              	{
              		temp_a1 += T[index + ref].a1;
                  	temp_a2 += T[index + ref].a2;
                  	temp_a3 += T[index + ref].a3;
                  	temp_b2 += T[index + ref].b2;
               		temp_b3 += T[index + ref].b3;
                	temp_c3 += T[index + ref].c3;
              	}
            	T_x[local_index].a1 = temp_a1 / (float)filter_size;
              	T_x[local_index].a2 = temp_a2 / (float)filter_size;
               	T_x[local_index].a3 = temp_a3 / (float)filter_size;
               	T_x[local_index].b2 = temp_b2 / (float)filter_size;
             	T_x[local_index].b3 = temp_b3 / (float)filter_size;
            	T_x[local_index].c3 = temp_c3 / (float)filter_size;
	}
	else return;
}

__global__
void filter_y_gpu(matrix *T_x, matrix *T_y, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	int index = (idz * width * height)+(idy * width)+idx;


	if(idx <= (width - half) && idy <= (height - filter_size) && idz <= (depth - half_z))
	{
		float temp_a1 = 0, temp_a2 = 0, temp_a3 = 0, temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;

              	int local_idy = idy + half;
            	int local_index = (idz * width * height) + (local_idy * width) + idx;

		for(int ref = 0; ref < filter_size; ref++)
              	{
            		temp_a1 += T_x[index + (ref * width)].a1;
                 	temp_a2 += T_x[index + (ref * width)].a2;
                  	temp_a3 += T_x[index + (ref * width)].a3;
                      	temp_b2 += T_x[index + (ref * width)].b2;
                    	temp_b3 += T_x[index + (ref * width)].b3;
                  	temp_c3 += T_x[index + (ref * width)].c3;
           	}
           	T_y[local_index].a1 = temp_a1 / filter_size;
          	T_y[local_index].a2 = temp_a2 / filter_size;
           	T_y[local_index].a3 = temp_a3 / filter_size;
           	T_y[local_index].b2 = temp_b2 / filter_size;
             	T_y[local_index].b3 = temp_b3 / filter_size;
          	T_y[local_index].c3 = temp_c3 / filter_size;
	}
	else return;
}

__global__
void filter_z_gpu(matrix *T_y, matrix *T_z, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	int index = (idz * width * height)+(idy * width)+idx;


	if(idx <= (width - half) && idy <= (height - half) && idz <= (depth - filter_size_z))
	{
		float temp_a1 = 0, temp_a2 = 0, temp_a3 = 0, temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;

              	int local_idz = idz + half_z;
            	int local_index = (local_idz * width * height) + (idy * width) + idx;

		for(int ref = 0; ref < filter_size_z; ref++)
              	{
            		temp_a1 += T_y[index + (ref * width * height)].a1;
                 	temp_a2 += T_y[index + (ref * width * height)].a2;
                  	temp_a3 += T_y[index + (ref * width * height)].a3;
                      	temp_b2 += T_y[index + (ref * width * height)].b2;
                    	temp_b3 += T_y[index + (ref * width * height)].b3;
                  	temp_c3 += T_y[index + (ref * width * height)].c3;
           	}
           	T_z[local_index].a1 = temp_a1 / filter_size_z;
          	T_z[local_index].a2 = temp_a2 / filter_size_z;
           	T_z[local_index].a3 = temp_a3 / filter_size_z;
           	T_z[local_index].b2 = temp_b2 / filter_size_z;
             	T_z[local_index].b3 = temp_b3 / filter_size_z;
          	T_z[local_index].c3 = temp_c3 / filter_size_z;
	}
	else return;
}

__global__
void remove_padding_gpu(matrix *T_z, matrix *T_m, int half, int half_z, int width, int height, int depth)
{

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;

	int index = (idz * width * height)+(idy * width)+idx;

	if(idx >= half && idx < (width - half) && idy >= half && idy < (height - half) && idz >= half_z && idz < (depth - half_z))
	{
		int local_idx = idx - half;
		int local_idy = idy - half;
		int local_idz = idz - half_z;
		int local_width = width - 2 * half;
		int local_height = height - 2 * half;
		int local_index = (local_idz * local_width * local_height) + (local_idy * local_width) + local_idx;	

		T_m[local_index].a1 = T_z[index].a1;
		T_m[local_index].a2 = T_z[index].a2;
		T_m[local_index].a3 = T_z[index].a3;
		T_m[local_index].b2 = T_z[index].b2;
		T_m[local_index].b3 = T_z[index].b3;
		T_m[local_index].c3 = T_z[index].c3;
	}
	else return;
}

void partial_derv(float *dx, float *dy, float *dz, int *pixel, int h, int h2, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	size_t shared_mem_size = 10 * 10 * 10 * sizeof(int);
	partial_derv_gpu<<<blocks, threads, shared_mem_size>>>(dx, dy, dz, pixel, h, h2, width, height, depth);
}


void tensor(matrix *T, float *dx, float *dy, float *dz, int size)
{
	dim3 threads(1024);
	dim3 blocks(ceil(size/1024.0));
	tensor_gpu<<<blocks, threads>>> (T, dx, dy, dz, size);
}

void filter_x(matrix *T, matrix *T_x, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	filter_x_gpu<<<blocks, threads>>>(T, T_x, filter_size, filter_size_z, half, half_z, width, height, depth);	
}

void filter_y(matrix *T_x, matrix *T_y, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	filter_y_gpu<<<blocks, threads>>>(T_x, T_y, filter_size, filter_size_z, half, half_z, width, height, depth);	
}

void filter_z(matrix *T_y, matrix *T_z, int filter_size, int filter_size_z, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	filter_z_gpu<<<blocks, threads>>>(T_y, T_z, filter_size, filter_size_z, half, half_z, width, height, depth);	
}

void remove_padding(matrix *T_z, matrix *T_m, int half, int half_z, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	remove_padding_gpu<<<blocks, threads>>>(T_z, T_m, half, half_z, width, height, depth);	
}
