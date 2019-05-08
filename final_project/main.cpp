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
#include <Eigen>
#include <omp.h>

using namespace std;

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))				// useful MACRO to check for errors
static void HandleError( cudaError_t err, const char *file,  int line ) {			// error handling function from Dr. Mayerich
   	if (err != cudaSuccess) {
	     cout<<cudaGetErrorString( err )<<" in "<< file <<" at line "<< line;
   	}
}


using namespace cimg_library;
using namespace Eigen;

void write_trace(vector<position> trace, int index)
{
	string filename = "./Trace/trace_" + to_string(index) + ".bin";
	ofstream trace_w(filename.c_str(), ios::binary);
	
	trace_w.write( (char *) &trace[0].x, (trace.size()) * 3 * sizeof(float));
	trace_w.close();	
}

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
	cout << "time taken to read files			: " << time.count()*1000 << " ms" << " 		CPU (using Cimg Library)" <<endl;

	int h = 1;
	int h2 = 4;
	long unsigned int size = pixel.size();

	cudaEvent_t begin, end;
	float gpu_time = 0.0f;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

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
	cout << "time taken to calculate derivates		: " << gpu_time <<" ms" << " 		GPU" << endl;

	cudaFree(pixel_gpu);
/*
	vector<float> dx(size,0);
	vector<float> dy(size,0);
	vector<float> dz(size,0);
	HANDLE_ERROR(cudaMemcpy(&dx[0], dx_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&dy[0], dy_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&dz[0], dz_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));
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
*/

// Calculate the tensors
//
	matrix *T_gpu;

	HANDLE_ERROR(cudaMalloc((void**) &T_gpu, size * sizeof(matrix)));

	cudaEventRecord(begin);
	tensor(T_gpu, dx_gpu, dy_gpu, dz_gpu, size);								// dx, dy, dz already resides in gpu memory.
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, begin, end);
	cout << "time taken to calculate tensors			: " << gpu_time <<" ms" << "		GPU" << endl;

	cudaFree(dx_gpu);
	cudaFree(dy_gpu);
	cudaFree(dz_gpu);
	
//	vector<matrix> T(size);
//	HANDLE_ERROR(cudaMemcpy(&T[0].a1, T_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_w("T.bin", ios::binary);
//	T_w.write((char *) &T[0].a1, size * sizeof(matrix));
//	T_w.close();

// Create a filter and pad the data
//
     	int filter_size = 16;
        int half = filter_size / 2;
 
        int filter_size_z = 4;
        int half_z = filter_size_z / 2; 
 
        width = width + 2 * half;
        height = height + 2 * half;
        depth = depth + 2 * half_z;
	size = width * height * depth;

	matrix *T_pad_gpu;
	HANDLE_ERROR(cudaMalloc((void**) &T_pad_gpu, size * sizeof(matrix)));

	cudaEventRecord(begin);
	padding(T_gpu, T_pad_gpu, half, half_z, width, height, depth);
	cudaEventRecord(end);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, begin, end);
	cout << "time taken to apply padding 			: " << gpu_time <<" ms" << "		GPU" << endl;

	cudaFree(T_gpu);

//	vector<matrix> T_pad(size);
//	HANDLE_ERROR(cudaMemcpy(&T_pad[0].a1, T_pad_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_pad_w("T_pad_gpu.bin", ios::binary);
//	T_pad_w.write((char *) &T_pad[0].a1, size * sizeof(matrix));
//	T_pad_w.close();

// apply filter along X direction
//	
	matrix *T_x_gpu, *T_y_gpu, *T_z_gpu;
	HANDLE_ERROR(cudaMalloc((void**) &T_x_gpu, size * sizeof(matrix)));

	cudaEventRecord(begin);
	filter_x(T_pad_gpu, T_x_gpu, filter_size, filter_size_z, half, half_z, width, height, depth);
	cudaEventRecord(end);
	
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, begin, end);
	cout << "time taken to apply blurring along x direc	: " << gpu_time <<" ms" << "		GPU-SHMEM" << endl;

	cudaFree(T_pad_gpu);

//	vector<matrix> T_x(size);
//	HANDLE_ERROR(cudaMemcpy(&T_x[0].a1, T_x_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_x_w("T_x_gpu.bin", ios::binary);
//	T_x_w.write((char *) &T_x[0].a1, size * sizeof(matrix));
//	T_x_w.close();

// apply filter along Y direction
//
	
	HANDLE_ERROR(cudaMalloc((void**) &T_y_gpu, size * sizeof(matrix)));

	cudaEventRecord(begin);
	filter_y(T_x_gpu, T_y_gpu, filter_size, filter_size_z, half, half_z, width, height, depth);
	cudaEventRecord(end);
	
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, begin, end);
	cout << "time taken to apply blurring along y direc	: " << gpu_time <<" ms" << "		GPU-SHMEM" << endl;

	cudaFree(T_x_gpu);

//	vector<matrix> T_y(size);
//	HANDLE_ERROR(cudaMemcpy(&T_y[0].a1, T_y_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_y_w("T_y_gpu.bin", ios::binary);
//	T_y_w.write((char *) &T_y[0].a1, size * sizeof(matrix));
//	T_y_w.close();

// apply filter along Z direction
//
	HANDLE_ERROR(cudaMalloc((void**) &T_z_gpu, size * sizeof(matrix)));

	cudaEventRecord(begin);
	filter_z(T_y_gpu, T_z_gpu, filter_size, filter_size_z, half, half_z, width, height, depth);
	cudaEventRecord(end);
	
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, begin, end);
	cout << "time taken to apply blurring along z direc	: " << gpu_time <<" ms" << "		GPU-SHMEM" <<endl;

	cudaFree(T_y_gpu);

//	vector<matrix> T_z(size);
//	HANDLE_ERROR(cudaMemcpy(&T_z[0].a1, T_z_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
	
//	ofstream T_z_w("T_z_gpu.bin", ios::binary);
//	T_z_w.write((char *) &T_z[0].a1, size * sizeof(matrix));
//	T_z_w.close();

// Remove padding
//
	int new_size = (width - 2 * half) * (height - 2 * half) * (depth - 2 * half_z);
	vector< matrix > T_m(new_size, {0, 0, 0, 0, 0, 0});
	matrix *T_m_gpu;

	HANDLE_ERROR(cudaMalloc((void**) &T_m_gpu, new_size * sizeof(matrix)));

	cudaEventRecord(begin);
	remove_padding(T_z_gpu, T_m_gpu, half, half_z, width, height, depth);
	cudaEventRecord(end);
	
	HANDLE_ERROR(cudaMemcpy(&T_m[0].a1, T_m_gpu, new_size * sizeof(matrix), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(end);

	cudaFree(T_z_gpu);
	cudaFree(T_m_gpu);

	cudaEventElapsedTime(&gpu_time, begin, end);
	cout << "time taken to remove padding			: " << gpu_time <<" ms" << "		GPU" << endl;

	size = new_size;
	width = width - 2 * half;
	height = height - 2 * half;
	depth = depth - 2 * half_z;

//	ofstream T_m_w("T_m_gpu.bin", ios::binary);
//	T_m_w.write((char *) &T_m[0].a1, size * sizeof(matrix));
//	T_m_w.close();

// move the data to a type suitable for eigen libarary
//
	vector<Matrix3f> T_mat(size);

	#pragma omp parallel for
	for(unsigned int i = 0; i < size; i++)
	{
		T_mat[i] << T_m[i].a1, T_m[i].a2, T_m[i].a3,
			    T_m[i].a2, T_m[i].b2, T_m[i].b3,
			    T_m[i].a3, T_m[i].b3, T_m[i].c3;
	}

// Compute Eigen Vectors

	vector<eigen_vectors>Evec(size, {0, 0, 0});

	int nthreads;
	double omp_start = omp_get_wtime();
	#pragma omp parallel for
	for (unsigned int ii = 0; ii < Evec.size(); ii++)	
	{
        	if(ii == 0) nthreads = omp_get_num_threads();
		SelfAdjointEigenSolver<Matrix3f> handle;
		handle.computeDirect(T_mat[ii]);
		Vector3f::Map(&Evec[ii].x) = handle.eigenvectors().col(0);
	}
	double omp_stop = omp_get_wtime();
	double omp_time = omp_stop - omp_start;

	cout << "time taken to calculate eigen vector		: " << omp_time * 1000<< " ms" << "		CPU-OPENMP(Used " << nthreads << " threads)" << endl;

//	ofstream Evec_w("Evec_gpu.bin", ios::binary);
//	Evec_w.write((char *) &Evec[0].x, size * sizeof(matrix));
//	Evec_w.close();

// do tractography (CPU IMPLEMENTATION)

	float del_t = 0.2;	
	const int n_steps = 10000;
/*
	vector< position > trace;
	int index;	
	omp_start = omp_get_wtime();
	#pragma omp parallel for private(index, trace) collapse(3)
	for(int aisle = 0; aisle < 5; aisle++)
	{
		for(int row = 0; row < 5; row++)
		{
			for(int col = 0; col < 5; col++)
			{
				index = ((aisle) * width * height) + ((row) * width) + (col);
				int tid=omp_get_thread_num();
        			if(tid==0 && index < 1)
				{
            				nthreads=omp_get_num_threads();
        			}
				trace.clear();

				eulers_method(&trace, Evec, col, row, aisle, width, height, depth, del_t, n_steps);
				write_trace(trace, index);
			}
		}
	}
	omp_stop = omp_get_wtime();
	omp_time = (omp_stop - omp_start);

	cout << "time taken to do cryptography and writing it	: " << omp_time * 1000<< " ms" << "		CPU-OPENMP(Used " << nthreads << " threads)" << endl;
*/

// do tractography (GPU IMPLEMENTATION) ***** DO NOT USE THIS (Needs better file handling and memory handling methodology)*****


	cout << "GPU Verison for tractography needs better memory handling and file handling methodology" << endl;
	vector< array<position, (n_steps+1)> > trace_g(5*5*5);
	
	position *trace_gpu;
	eigen_vectors *Evec_gpu;	

	HANDLE_ERROR(cudaMalloc((void**) &trace_gpu, 5 * 5 * 5 * (n_steps+1) * sizeof(position)));
	HANDLE_ERROR(cudaMalloc((void**) &Evec_gpu, size * sizeof(eigen_vectors)));

	HANDLE_ERROR(cudaMemcpy(Evec_gpu, &Evec[0].x, size * sizeof(eigen_vectors), cudaMemcpyHostToDevice));

	cudaEventRecord(begin);
	tractography(trace_gpu, Evec_gpu, del_t, n_steps, 5, 5, 5);
	cudaEventRecord(end);
		
	HANDLE_ERROR(cudaMemcpy(&trace_g[0][0].x, trace_gpu, 5 * 5 * 5 * (n_steps + 1) * sizeof(position), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(end);
	
	cudaFree(Evec_gpu);
	cudaFree(trace_gpu);

	return 0;
}
