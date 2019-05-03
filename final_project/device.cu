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

void partial_derv(float *dx, float *dy, float *dz, int *pixel, int h, int h2, int width, int height, int depth)
{
	dim3 threads(8,8,8);
	dim3 blocks(ceil(width/8.0f), ceil(height/8.0f), ceil(depth/8.0f));
	std::cout << "depth : " << 8*ceil(depth/8.0f) << std::endl;
	std::cout << "width : " << 8*ceil(width/8.0f) << std::endl;
	size_t shared_mem_size = 10 * 10 * 10 * sizeof(int);
	partial_derv_gpu<<<blocks, threads, shared_mem_size>>>(dx, dy, dz, pixel, h, h2, width, height, depth);
}


void tensor(matrix *T_gpu, float *dx, float *dy, float *dz, int size)
{
	dim3 threads(1024);
	dim3 blocks(ceil(size/1024.0));
	tensor_gpu<<<blocks, threads>>> (T_gpu, dx, dy, dz, size);
}	
