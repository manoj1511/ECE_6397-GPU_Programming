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

int main()
{				
	vector<string> filenames;								// An array to store filenames
	string pattern = "./kidney_100/test_*.jpg";						// File pattern to be read
	glob_t glob_result;									// A glob handle
	memset(&glob_result, 0, sizeof(glob_result));						// make it to 0
	int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);		// read the files into glob_result	
	if(return_value != 0)									// error checker	
	{
		globfree(&glob_result);
		stringstream ss;
		ss << "glob() failed with return_value " << return_value << endl;
		throw runtime_error(ss.str());
	}

	for(unsigned int i = 0; i < glob_result.gl_pathc; i++)					// iterate the files
	{
		filenames.push_back(string(glob_result.gl_pathv[i]));				// and store it in the vector
	}
	globfree(&glob_result);									// free the glob

	cout << filenames.size()<< " files" <<endl;						// display number of files
	cout << endl;
	int depth = filenames.size();								// num of files will be the depth

	vector<int> pixel;									// vector of int's to store image pixel values
	int width = 0;										// Declare variables to store width and height and initialize to 0
	int height = 0;

	chrono::high_resolution_clock::time_point start,stop;					// Declare the cpu timers
	start = chrono::high_resolution_clock::now();						// record start
	for(auto &file : filenames)								// iterate the files
	{	
		cimg_library::CImg<unsigned char> image(file.c_str());				// read files one by one using CImg library
		width = image.width();								// assign the width and height
		height = image.height();
		for( auto &it : image)								// iterate through the pixels in image
			pixel.push_back((int)it);						// store it in vector pixel
	}
	stop = chrono::high_resolution_clock::now();						// record stop
	chrono::duration<double> time;								// declare time to store the time taken	
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);			// Calculate the time taken
	cout << "time taken to read files			: " << time.count()*1000 << " ms" << " 		CPU (using Cimg Library)" <<endl;// Display the time taken

// Calculate partial derivatives
//
	int h = 1;										// Step size for X and Y direction
	int h2 = 4;										// Step size along Z diretion
	long unsigned int size = pixel.size();							// Assign the size
	cudaEvent_t begin, end;									// Declare events for GPU timer
	float gpu_time = 0.0f;									// Declare a variable to store time
	cudaEventCreate(&begin);								// Create the events
	cudaEventCreate(&end);
	float *dx_gpu, *dy_gpu, *dz_gpu; 							// Declare the gpu pointers for dx, dy, dz and pizel
	int *pixel_gpu;

	HANDLE_ERROR(cudaMalloc((void**) &dx_gpu, size * sizeof(float)));			// Allocate memory for dx, dy, dx and pixel on gpu
	HANDLE_ERROR(cudaMalloc((void**) &dy_gpu, size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**) &dz_gpu, size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**) &pixel_gpu, size * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(pixel_gpu, &pixel[0], size * sizeof(int), cudaMemcpyHostToDevice));	// Copy pixel to gpu

	cudaEventRecord(begin);									// put begin event on stream
	partial_derv(dx_gpu, dy_gpu, dz_gpu, pixel_gpu, h, h2, width, height, depth);		// call the function that launches GPU kernel
	cudaEventRecord(end);									// put end event on stream

	cudaEventSynchronize(end);								// Synchronize the end		
	cudaEventElapsedTime(&gpu_time, begin, end);						// Calculate the time taken
	cout << "time taken to calculate derivates		: " << gpu_time <<" ms" << " 		GPU" << endl;	// Display the time

	cudaFree(pixel_gpu);									// free up GPU memory of pixel
/*
	vector<float> dx(size,0);								// Uncomment and use this block to save the dx, dy, dz values in binary format
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
	matrix *T_gpu;										// Create GPU pointer of type matrix for structure tensors

	HANDLE_ERROR(cudaMalloc((void**) &T_gpu, size * sizeof(matrix)));			// Allocate memory for structure tensors

	cudaEventRecord(begin);									// Send begin event to stream
	tensor(T_gpu, dx_gpu, dy_gpu, dz_gpu, size);						// dx, dy, dz already in gpu memory. Call the function that calls the kernel
	cudaEventRecord(end);									// Send end event to stream

	cudaEventSynchronize(end);								// synchronize for end
	cudaEventElapsedTime(&gpu_time, begin, end);						// Calculate the time taken
	cout << "time taken to calculate tensors			: " << gpu_time <<" ms" << "		GPU" << endl;	// Display the time taken

	cudaFree(dx_gpu);									// free up dx, dy, dz on GPU
	cudaFree(dy_gpu);
	cudaFree(dz_gpu);
	
//	vector<matrix> T(size);									// Uncomment this to write the value of tensors
//	HANDLE_ERROR(cudaMemcpy(&T[0].a1, T_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_w("T.bin", ios::binary);
//	T_w.write((char *) &T[0].a1, size * sizeof(matrix));
//	T_w.close();

// Create a filter and pad the data
//
     	int filter_size = 16;									// Declare filter size along X and Y direction
        int half = filter_size / 2;								// Calculate half of it
        int filter_size_z = 4;									// Declare filter size along Z direction
        int half_z = filter_size_z / 2; 							// Calculate half of it
 
        width = width + 2 * half;								// increase the width and height and depth for padding
        height = height + 2 * half;
        depth = depth + 2 * half_z;
	size = width * height * depth;								// Calculate the new size including the halo regions

	matrix *T_pad_gpu;									// Pointer to padded tensor on GPU
	HANDLE_ERROR(cudaMalloc((void**) &T_pad_gpu, size * sizeof(matrix)));			// Allcoate memory on GPU

	cudaEventRecord(begin);									// Send begin to stream
	padding(T_gpu, T_pad_gpu, half, half_z, width, height, depth);				// Call the function that calls the padding GPU kernel
	cudaEventRecord(end);									// Send end to stream

	cudaEventSynchronize(end);								// Synchronize the end
	cudaEventElapsedTime(&gpu_time, begin, end);						// Calculate the time taken
	cout << "time taken to apply padding 			: " << gpu_time <<" ms" << "		GPU" << endl;	// Display the time taken

	cudaFree(T_gpu);									// Free up unpadded tensor data on GPU

//	vector<matrix> T_pad(size);								// Use this block to write padded tensors
//	HANDLE_ERROR(cudaMemcpy(&T_pad[0].a1, T_pad_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_pad_w("T_pad_gpu.bin", ios::binary);
//	T_pad_w.write((char *) &T_pad[0].a1, size * sizeof(matrix));
//	T_pad_w.close();

// apply filter along X direction
//	
	matrix *T_x_gpu, *T_y_gpu, *T_z_gpu;							// Declare GPU pointers for blurring
	HANDLE_ERROR(cudaMalloc((void**) &T_x_gpu, size * sizeof(matrix)));			// Allocate for blurring along X direction

	cudaEventRecord(begin);									// Send begin to stream
	filter_x(T_pad_gpu, T_x_gpu, filter_size, filter_size_z, half, half_z, width, height, depth);	// Call the function that calls the gpu kernel
	cudaEventRecord(end);									// send end to stream
	
	cudaEventSynchronize(end);								// Synchronize the data
	cudaEventElapsedTime(&gpu_time, begin, end);						// Calculate the time taken
	cout << "time taken to apply blurring along x direc	: " << gpu_time <<" ms" << "		GPU-SHMEM" << endl;	// Display the time taken

	cudaFree(T_pad_gpu);									// Free up padded GPU data

//	vector<matrix> T_x(size);								// Use to write the blur along x data
//	HANDLE_ERROR(cudaMemcpy(&T_x[0].a1, T_x_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_x_w("T_x_gpu.bin", ios::binary);
//	T_x_w.write((char *) &T_x[0].a1, size * sizeof(matrix));
//	T_x_w.close();

// apply filter along Y direction
//
	
	HANDLE_ERROR(cudaMalloc((void**) &T_y_gpu, size * sizeof(matrix)));			// Allocate for blurring along Y direction

	cudaEventRecord(begin);									// Send begin to stream
	filter_y(T_x_gpu, T_y_gpu, filter_size, filter_size_z, half, half_z, width, height, depth);	// call function that calls the kernel
	cudaEventRecord(end);									// Send end to stream
					
	cudaEventSynchronize(end);								// synchronize at end
	cudaEventElapsedTime(&gpu_time, begin, end);						// Calculate the time taken
	cout << "time taken to apply blurring along y direc	: " << gpu_time <<" ms" << "		GPU-SHMEM" << endl;	// Display the time taken

	cudaFree(T_x_gpu);									// Free up blur along X data

//	vector<matrix> T_y(size);								// Use this to write Blur along Y data
//	HANDLE_ERROR(cudaMemcpy(&T_y[0].a1, T_y_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_y_w("T_y_gpu.bin", ios::binary);
//	T_y_w.write((char *) &T_y[0].a1, size * sizeof(matrix));
//	T_y_w.close();

// apply filter along Z direction
//
	HANDLE_ERROR(cudaMalloc((void**) &T_z_gpu, size * sizeof(matrix)));			// Allocate memory on GPU for blurring along Y direction

	cudaEventRecord(begin);									// Send begin to stream
	filter_z(T_y_gpu, T_z_gpu, filter_size, filter_size_z, half, half_z, width, height, depth);	// Call funtion that calls the GPU kernel
	cudaEventRecord(end);									// Send end to stream
	
	cudaEventSynchronize(end);								// Synchronize for end
	cudaEventElapsedTime(&gpu_time, begin, end);						// Calculate the time taken
	cout << "time taken to apply blurring along z direc	: " << gpu_time <<" ms" << "		GPU-SHMEM" <<endl;	// Apply blurring along Z direction

	cudaFree(T_y_gpu);									// Free up blur data along Y direction

//	vector<matrix> T_z(size);								// Use this to write Bluring along Z to memory
//	HANDLE_ERROR(cudaMemcpy(&T_z[0].a1, T_z_gpu, size * sizeof(matrix), cudaMemcpyDeviceToHost));
//	ofstream T_z_w("T_z_gpu.bin", ios::binary);
//	T_z_w.write((char *) &T_z[0].a1, size * sizeof(matrix));
//	T_z_w.close();

// Remove padding
//
	int new_size = (width - 2 * half) * (height - 2 * half) * (depth - 2 * half_z);		// size after padding is removed
	vector< matrix > T_m(new_size, {0, 0, 0, 0, 0, 0});					// Allocate vector of matrix type
	matrix *T_m_gpu;									// Pointer to unpadded blurred tensor on GPU

	HANDLE_ERROR(cudaMalloc((void**) &T_m_gpu, new_size * sizeof(matrix)));			// Allocate memory on GPU

	cudaEventRecord(begin);									// Send begin to stream
	remove_padding(T_z_gpu, T_m_gpu, half, half_z, width, height, depth);			// Call the function that calls the kernel
	cudaEventRecord(end);									// send end to stream
	
	HANDLE_ERROR(cudaMemcpy(&T_m[0].a1, T_m_gpu, new_size * sizeof(matrix), cudaMemcpyDeviceToHost));	// Copy back unpadded blurred tensor to CPU
	cudaEventSynchronize(end);								// Synchronize for end

	cudaFree(T_z_gpu);									// Free up padded blurred data
	cudaFree(T_m_gpu);									// Free up unpadded blurred data as I have copied back to CPU

	cudaEventElapsedTime(&gpu_time, begin, end);						// Calculate the time taken
	cout << "time taken to remove padding			: " << gpu_time <<" ms" << "		GPU" << endl;	// Display the time taken

	size = new_size;									// Assign the current size
	width = width - 2 * half;								// Change back other dimention variables as well
	height = height - 2 * half;
	depth = depth - 2 * half_z;

//	ofstream T_m_w("T_m_gpu.bin", ios::binary);						// use to write the unpadded blurred tensor to gpu
//	T_m_w.write((char *) &T_m[0].a1, size * sizeof(matrix));
//	T_m_w.close();

// move the data to a type suitable for eigen libarary
//
	vector<Matrix3f> T_mat(size);								// create a vector of Eigen::Matrix type. Used for later processing
	#pragma omp parallel for								// OpenMP acceleration
	for(unsigned int i = 0; i < size; i++)							// Iterate throught the entire data
	{
		T_mat[i] << T_m[i].a1, T_m[i].a2, T_m[i].a3,					// Copy it to the Eigen::Matrix type vector
			    T_m[i].a2, T_m[i].b2, T_m[i].b3,					// I am doing this here because Eigen datatypes are not fully supported in GPU
			    T_m[i].a3, T_m[i].b3, T_m[i].c3;
	}

// Compute Eigen Vectors

	vector<eigen_vectors>Evec(size, {0, 0, 0});						// Create a vector of Eigen_vectors type
	cout << "computing Eigen vectors accelerated using OpenMP" << endl;			
	int nthreads;										// Used to store num of threads used in for loop
	double omp_start = omp_get_wtime();							// record the start
	#pragma omp parallel for								// Accelerate using OpenMP
	for (unsigned int ii = 0; ii < Evec.size(); ii++)					// Iterate through the structure tensor
	{
        	if(ii == 0) nthreads = omp_get_num_threads();					// write the num of threads used in first iteration
		SelfAdjointEigenSolver<Matrix3f> handle;					// Crate a handlefor the EigenSolver
		handle.computeDirect(T_mat[ii]);						// Compute the Eigen vectors
		Vector3f::Map(&Evec[ii].x) = handle.eigenvectors().col(0);			// Mapt the first eigen vector to vector of my oen eigen_vector type
	}
	double omp_stop = omp_get_wtime();							// Record end event
	double omp_time = omp_stop - omp_start;							// Calculate the time taken

	cout << "time taken to calculate eigen vector		: " << omp_time * 1000<< " ms" << "		CPU-OPENMP(Used " << nthreads << " threads)" << endl;	// Display the time taken

//	ofstream Evec_w("Evec_gpu.bin", ios::binary);						// Use this to write Eigen vectors to memory
//	Evec_w.write((char *) &Evec[0].x, size * sizeof(matrix));
//	Evec_w.close();

// do tractography 

	float del_t = 0.2;									// Initialize time interval for tractography
	const int n_steps = 100000;								// Initialze num of time steps

	vector< array<position, (n_steps+1)> > trace_g(5*5*5);					// Create Vector of arrays. This is contiguos in memory. Vector of vector is not
	
	position *trace_gpu;									// GPU pointer to trace
	eigen_vectors *Evec_gpu;								// GPU pointer for eigen vectors

	HANDLE_ERROR(cudaMalloc((void**) &trace_gpu, 5 * 5 * 5 * (n_steps+1) * sizeof(position)));	// Allcate memory for trace in GPU
	HANDLE_ERROR(cudaMalloc((void**) &Evec_gpu, size * sizeof(eigen_vectors)));		// Allocate memory for eigen vectors

	HANDLE_ERROR(cudaMemcpy(Evec_gpu, &Evec[0].x, size * sizeof(eigen_vectors), cudaMemcpyHostToDevice));	// Copy Eigen vector data to GPU

	cudaEventRecord(begin);									// Send begin to stream
	tractography(trace_gpu, Evec_gpu, del_t, n_steps, 5, 5, 5);				// Call the function that calls tractography kernel on GPU
	cudaEventRecord(end);									// Send the end to stream
		
	cudaEventSynchronize(end);								// Synchronize the end
	cudaEventElapsedTime(&gpu_time, begin, end);						// Calculate the time taken
	cout << "time taken to do tractography			: " << gpu_time <<" ms" << "		GPU" << endl;	// Display the time taken

	HANDLE_ERROR(cudaMemcpy(&trace_g[0][0].x, trace_gpu, 5 * 5 * 5 * (n_steps + 1) * sizeof(position), cudaMemcpyDeviceToHost));	// Copy back the traces to CPU

	cudaFree(Evec_gpu);									// Free the Eigen vector GPU memory
	cudaFree(trace_gpu);									// Free the trace GPU memory
	cout << endl;										
	cout << "GPU Verison for tractography needs better memory handling and file handling methodology" << endl;
	cout << "I'm not writing trace. I'm using the same function as both __device__ and __host__ function " << endl;
	return 0;
}
