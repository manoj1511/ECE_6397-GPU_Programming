// Author : Manoj Kumar Cebol Sundarrajan
// Organization : University of Houston

#include <iostream>
#include <vector>
#include <glob.h>
#include <string>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include "CImg.h"
#ifdef Success
  #undef Success
#endif
#include <chrono>
#include <fstream>
#include "device.cuh"

using namespace std;

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))				// useful MACRO to check for errors
static void HandleError( cudaError_t err, const char *file,  int line ) {			// error handling function from Dr. Mayerich
   	if (err != cudaSuccess) {
	     cout<<cudaGetErrorString( err )<<" in "<< file <<" at line "<< line;
   	}
}


using namespace cimg_library;

int main()
{

	vector<string> filenames;
	string pattern = "./kidney_100/test_00*.jpg";
	glob_t glob_result;
	memset(&glob_result, 0, sizeof(glob_result));
	int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
	if(return_value != 0)
	{
		globfree(&glob_result);
		stringstream ss;
		ss << "glob() failed with return_value " << return_value << endl;
		throw runtime_error(ss.str());
	}

	for(unsigned int i = 0; i < glob_result.gl_pathc; i++)
	{
		filenames.push_back(string(glob_result.gl_pathv[i]));
	}
	globfree(&glob_result);

	cout << filenames.size()<<endl;
	cout << endl;
	int depth = filenames.size();

	vector<int> pixel;
	int width = 0;
	int height = 0;

	chrono::high_resolution_clock::time_point start,stop;
	start = chrono::high_resolution_clock::now();

	for(auto &file : filenames)
	{	
		cimg_library::CImg<unsigned char> image(file.c_str());

		width = image.width();
		height = image.height();
		for( auto &it : image)
			pixel.push_back((int)it);
		
	}

	stop = chrono::high_resolution_clock::now();
	
	chrono::duration<double> time;					
	
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to read files				: " << time.count()*1000 << " ms" << endl;

	int h = 1;
	int h2 = 4;
	long unsigned int size = pixel.size();

	cudaEvent_t begin, end;
	float gpu_time = 0.0f;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	vector<float> dx(size,0);
	vector<float> dy(size,0);
	vector<float> dz(size,0);
	float *dx_gpu, *dy_gpu, *dz_gpu; 
	int *pixel_gpu;

	HANDLE_ERROR(cudaMalloc((void**) &dx_gpu, size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**) &dy_gpu, size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**) &dz_gpu, size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**) &pixel_gpu, size * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(pixel_gpu, &pixel[0], size * sizeof(int), cudaMemcpyHostToDevice));

	cudaEventRecord(begin);
	partial_derv(dx_gpu, dy_gpu, dz_gpu, pixel_gpu, h, h2, width, height, depth);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, begin, end);
	cout << "GPU Partial derivatives calculation time taken		: " << gpu_time <<" ms" << endl;

	HANDLE_ERROR(cudaMemcpy(&dx[0], dx_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&dy[0], dy_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&dz[0], dz_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(pixel_gpu);

	ofstream part_dx("dx.bin", ios::binary);
	part_dx.write((char *) &dx[0], size * sizeof(float));
	part_dx.close();

	ofstream part_dy("dy.bin", ios::binary);
	part_dy.write((char *) &dy[0], size * sizeof(float));
	part_dy.close();

	ofstream part_dz("dz.bin", ios::binary);
	part_dz.write((char *) &dz[0], size * sizeof(float));
	part_dz.close();

	ofstream pix_w("pixel.bin", ios::binary);
	pix_w.write((char *) &pixel[0], size * sizeof(int));
	pix_w.close();

	vector<matrix> T(size, {0,0,0,0,0,0,0,0,0});
	matrix *T_gpu;

	HANDLE_ERROR(cudaMalloc((void**) &T_gpu, size * sizeof(matrix)));

	cudaEventRecord(begin);
	tensor(T_gpu, dx_gpu, dy_gpu, dz_gpu, size);								// dx, dy, dz already resides in gpu memory.
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, begin, end);
	cout << "GPU Tensor calculation time taken			: " << gpu_time <<" ms" << endl;

	HANDLE_ERROR(cudaMemcpy(&T[0].a1, T_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));

	cudaFree(dx_gpu);
	cudaFree(dy_gpu);
	cudaFree(dz_gpu);
	
	ofstream T_w("T.bin", ios::binary);
	T_w.write((char *) &T[0].a1, size * sizeof(matrix));
	T_w.close();

	return 0;
}
