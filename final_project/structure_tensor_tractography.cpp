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
//#include<mkl_lapacke.h>
//#define EIGEN_USE_MKL_ALL
#include <Eigen>
#include <cmath>
#include <fstream>
#include <omp.h>

#define pi 3.141
#define mu 0


using namespace std;
using namespace cimg_library;
using namespace Eigen;

struct matrix										// Create my own matrix type
{
	float a1, a2, a3;
	float  	  b2, b3;
	float 	      c3;
};


struct eigen_vectors									// My own eigen vector type
{	
	float a, b, c;
};

struct pixels										// My own pixel type
{
	float r, g, b;
};

struct position										// My own position type
{
	float x,y,z;
};

void eulers_method(vector<position> *trace, vector<eigen_vectors> Evec, 
		   int col, int row, int aisle, int width, int height, int depth, 
		   float del_t, int n)							// Eulers method of integration to do tractography
{
	position initial = {(float) col, (float) row, (float) aisle};			// set the initial position based on col, row and aisle
	position new_pos = {0.0f, 0.0f, 0.0f};						// Declare a variable new_pos to store the updated position

	int init;									// init is the 1D index for 3D coordinates
	(*trace).push_back(initial);							// Push initial value to trace
	int step = 0;									// make step to 0. This will keep track of iteration
	while(step < n)									// iterate as long as step is less than n
	{				
		init = ((int)initial.z * width * height) + ((int)initial.y * width) + ((int)initial.x);	// Calculate the 1D index
		new_pos.x = initial.x + del_t * (Evec[init].a);				// Calculate the new pos along x Euler's method
		new_pos.y = initial.y + del_t * (Evec[init].b);				// Calculate the new pos along y Euler's mathod
		new_pos.z = initial.z + del_t * (Evec[init].c);				// Calculate the new pos along z Euler's method

		if( (new_pos.x < width) && (new_pos.y < height) && (new_pos.z < depth) && (new_pos.x >= 0) && (new_pos.y >= 0) && (new_pos.z >= 0) && 	  
		    (initial.x < width) && (initial.y < height) && (initial.z < depth) && (initial.x >= 0) && (initial.y >= 0) && (initial.z >= 0) )		
											// Proceed only if new_pos and initial are in bounds
		{
			int new_p  = ((int)new_pos.z * width * height) + ((int)new_pos.y * width) + ((int)new_pos.x);	
											// 1D index of new position 3D
			float dot_p = (Evec[new_p].a * Evec[init].a) +			
				      (Evec[new_p].b * Evec[init].b) +  
				      (Evec[new_p].c * Evec[init].c);			// Calculate the dot product of prev vector and current vector 
			if(dot_p < 0)							// check if dot product is negative
			{
				new_pos.x = initial.x + del_t * (-Evec[init].a);	// If yes, reverse the vector direction and calculate the new positon
				new_pos.y = initial.y + del_t * (-Evec[init].b);	// along x, y and z
				new_pos.z = initial.z + del_t * (-Evec[init].c);
			}
			(*trace).push_back(new_pos);					// push the new_pos to trace
			initial.x = new_pos.x;						// update the intial coordinates with new position
			initial.y = new_pos.y;
			initial.z = new_pos.z;
			step++;								// increment the step				
		}
		else break;								// break if the positions are out of bounds
	}
	return;										// return from the function
}

void write_trace(vector<position> trace, int index)					// A function to write the trace into a file
{
	string filename = "./Trace/trace_" + to_string(index) + ".bin";			// Write to the folder Trace and the file name is based on index
	ofstream trace_w(filename.c_str(), ios::binary);				// create a file handle for the file. Open in binary mode
	trace_w.write( (char *) &trace[0].x, (trace.size()) * 3 * sizeof(float));	// Write from the first byte of the trace. Write the entire content
	trace_w.close();								// Close the file handle
}

int main()
{

	vector<string> filenames;							// an array to store filenames
	string pattern = "./kidney_100/test_*.jpg";					// The file pattern to read
	glob_t glob_result;								// glob object
	memset(&glob_result, 0, sizeof(glob_result));
	int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);	// read the filenames to glob_result
	if(return_value != 0)								// An error checker if it cannot read the files
	{
		globfree(&glob_result);
		stringstream ss;
		ss << "glob() failed with return_value " << return_value << endl;
		throw runtime_error(ss.str());
	}

	for(unsigned int i = 0; i < glob_result.gl_pathc; i++)				// iterate the filenames
	{
		filenames.push_back(string(glob_result.gl_pathv[i]));			// push it to the vector filenames
	}
	globfree(&glob_result);								// free the glob object

	cout << filenames.size() << " files" << endl;					// output the number of files read
	cout << endl;
	int depth = filenames.size();							// num of files will be the depth
	vector<int> pixel;								// create a vector of int's to store the image pixels
	int width, height;								// declare variables to store width and height
	chrono::high_resolution_clock::time_point start,stop;				// declare the timer's from chrono library
	start = chrono::high_resolution_clock::now();					// record the start point
	for(auto &file : filenames)							// iterate the filenames
	{	
		cimg_library::CImg<unsigned char> image(file.c_str());			// read the particular image file using CImg
		width = image.width();							// assign the width and height of the file to variables
		height = image.height();
		for( auto &it : image)							// iterate the image pixels
			pixel.push_back((int)it);					// pust the values to vector pixel
	}
	stop = chrono::high_resolution_clock::now();					// record the stop event
	chrono::duration<double> time;							// declare a variable to hold the time taken
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);		// calculate the time taken start - stop
	cout << "time taken to read files			: " << time.count()*1000 << " ms" << endl;	// Output the time taken to read images

	float h = 1;									// declare step size along x and y axis
	float h2 = 4;									// declare step size along z axis
	unsigned int size = pixel.size();						// declare the size as num of pixels read

// Calculate partial derivates
//
	vector<float> dx(size,0);							// allcoate vectors dx, dy, dz to store derivatives
	vector<float> dy(size,0);
	vector<float> dz(size,0);

	start = chrono::high_resolution_clock::now();					// record the start event
	for(unsigned int i = 0 ; i < size; i++)						// iterate the pixels
	{
		if(i % width == 0) dx[i] = (pixel[i+1] - pixel[i])/h;			// If im in first column use forward difference formula
		else if(i % width == (unsigned int)(width-1)) dx[i] = (pixel[i] - pixel[i-1])/h;	// If im in last column use backward difference formula
		else dx[i] = (pixel[i+1] - pixel[i-1])/(2*h);				// Elsewhere use centered difference formula
	}	
	for(unsigned int i = 0; i < size; i++)						// Iterate the pixels
	{	
		int r = i/width;							// Calculate the row I'm in
		if(r % height == 0) dy[i] = (pixel[width+i] - pixel[i])/h;		// if Im in first row use forward difference formula
		else if(r % height == height - 1) dy[i] = (pixel[i] - pixel[i-width])/h;// If Im in second row use backward difference formula
		else dy[i] = (pixel[i+width] - pixel[i-width])/(2*h);			// Elsewhere use central deifference formula
	}
	for(unsigned int i = 0; i < size; i++)						// Iterate through the pixels
	{
		if(i < (unsigned int)(height*width)) dz[i] = (pixel[i + (width*height)] - pixel[i])/h2;	// If I'm in first aisle use forward difference formula
		else if(i >= (size - (height*width))) dz[i] = (pixel[i] - pixel[i - (width*height)])/h2;// If I'm in Second aisle use the backward difference formula
		else dz[i] = (pixel[i + (width*height)] - pixel[i - (width*height)])/(2*h2);	// Elsewhere use the central difference formula
	}
	stop = chrono::high_resolution_clock::now();					// record the stop event
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);		// Calculate the time taken
	cout << "time taken to calculate partial derivatives	: " << time.count()*1000 << " ms" << endl;	// Output the time taken

// Calculate Structure tensor
//
	vector<matrix> T(size,{0,0,0,0,0,0});						// Allocate a vector of matrix type to store the structure tensors
	start = chrono::high_resolution_clock::now();					// record start event
	for (unsigned int i = 0; i < size; i++)						// iterate throught the deivatives
	{
		T[i].a1 = dx[i] * dx[i];						// calculate the tensors a1, a2, a3, b2, b3, c3
		T[i].a2 = dx[i] * dy[i];
		T[i].a3 = dx[i] * dz[i];
		T[i].b2 = dy[i] * dy[i];
		T[i].b3 = dy[i] * dz[i];
		T[i].c3 = dz[i] * dz[i];
	}
	stop = chrono::high_resolution_clock::now();					// record the stop event
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);		// calculate the time taken
	cout << "time taken to calculate Tensor field		: " << time.count()*1000 << " ms" << endl;	// Output the time taken

	dx.clear();									// Clear out the derivates to free space
	dy.clear();
	dz.clear();

// PADDING THE TENSOR FIELD FOR BLURRING
//
	int filter_size = 16;								// initialze filter size along x and y direction
        int half = filter_size / 2;							// calculate the half of it

	int filter_size_z = 4;								// initialize the filter size along z direction
        int half_z = filter_size_z / 2;							// Calculate the half of it

	width = width + 2 * half;							// Increase the width, height and depth based on the filter size
        height = height + 2 * half;
        depth = depth + 2 * half_z;
        vector<matrix> T_pad(width * height * depth);					// Allocate the matrix for new size

	int index;									// Variable to store index in 1D
        int local_row, local_col, local_aisle, local_width, local_height, local_index; 	// Local variables used to map the index to local index
	
	start = chrono::high_resolution_clock::now();					// record the start event
	for(int aisle = 0; aisle < depth; aisle++)					// aisle iterates the depth
        {
                for(int row = 0; row < height; row++)					// row iterates the height
                {
                        for(int col = 0; col < width; col++)				// col iterates the width
                        {
                                index = (aisle * width * height) + (row * width) + col;	// calulate the index in the padded array
                                local_col = col - half;					// map to the unpadded array
                                local_row = row - half;
                                local_aisle = aisle - half_z;
                                local_width = width - 2 * half;
                                local_height = height - 2 * half;
                                local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;// Calculate local_index of the unpadded array

                                if(col >= half && col < (width - half) && row >= half && row < (height - half) && aisle >= half_z && aisle < (depth - half_z))	
											// If indices fall along the local indices if unpadded array
                                {
                                        T_pad[index].a1 = T[local_index].a1;		// copy the data
                                        T_pad[index].a2 = T[local_index].a2;
                                        T_pad[index].a3 = T[local_index].a3;
                                        T_pad[index].b2 = T[local_index].b2;
                                        T_pad[index].b3 = T[local_index].b3;
                                        T_pad[index].c3 = T[local_index].c3;
                                }
				else							// else
				{
                                        T_pad[index].a1 = 0;				// make the halo's zero
                                        T_pad[index].a2 = 0;
                                        T_pad[index].a3 = 0;
                                        T_pad[index].b2 = 0;
                                        T_pad[index].b3 = 0;
                                        T_pad[index].c3 = 0;
				}
                        }
                }
        }
	stop = chrono::high_resolution_clock::now();					// record stop event
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);		// Calculate the time taken
	cout << "time taken to apply Padding			: " << time.count()*1000 << " ms" << endl;	// Display the time taken

	T.clear();									// Clear the unpadded Tensor array

// Apply the filter along X axis
//	
	vector<matrix> T_x(width * height * depth);					// Allocate a vector to store the tensors after blurring along X direction
	float temp_a1, temp_a2, temp_a3, temp_b2, temp_b3, temp_c3;			// Declare temp variables

	start = chrono::high_resolution_clock::now();					// record start event
	for(int aisle = 0; aisle < depth; aisle++)					// iterate the depth
        {
                for(int row = 0; row < height; row++)					// iterate the height
                {
                        for(int col = 0; col <= width - filter_size; col++)		// iterate the width. stop at a point so that the filter could cover the rest
                        {
                                temp_a1 = 0, temp_a2 = 0, temp_a3 = 0; 
				temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;			// make temps to 0
                                index = (aisle * width * height) + (row * width) + col;	// Calculate the index
                                local_col = col + half;					// offset col so that i donot write at halos
                                local_row = row;
                                local_aisle = aisle;
                                local_width = width;
                                local_height = height;
                                local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;	// calculte the index to write to

                                for(int ref = 0; ref < filter_size; ref++)		// iterate the filter number of times 
                                {
                                        temp_a1 += T_pad[index + ref].a1;		// sum up the values in temp
                                        temp_a2 += T_pad[index + ref].a2;
                                        temp_a3 += T_pad[index + ref].a3;
                                        temp_b2 += T_pad[index + ref].b2;
                                        temp_b3 += T_pad[index + ref].b3;
                                        temp_c3 += T_pad[index + ref].c3;
                                }
                                T_x[local_index].a1 = temp_a1 / filter_size;		// normalize using the filter size
                                T_x[local_index].a2 = temp_a2 / filter_size;		// and store it in T_x
                                T_x[local_index].a3 = temp_a3 / filter_size;
                                T_x[local_index].b2 = temp_b2 / filter_size;
                                T_x[local_index].b3 = temp_b3 / filter_size;
                                T_x[local_index].c3 = temp_c3 / filter_size;
                        }
                }
        }
	stop = chrono::high_resolution_clock::now();					// record the stop event
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);		// calculate the time taken
	cout << "time taken to apply blur along x axis		: " << time.count()*1000 << " ms" << endl;	// Display the time taken

	T_pad.clear();									// Clear out the memory of initial T_pad

// Apply the filter along Y axis
//
	vector<matrix> T_y(width * height * depth);					// Allocate a vector to store blurred tensors along y
	start = chrono::high_resolution_clock::now();					// start the timer
	for(int aisle = 0; aisle < depth; aisle++)					// iterate through depth
        {
                for(int row = 0; row <= height - filter_size; row++)			// iterate through height. stop at a point so the filter can cover the rest
                {
                        for(int col = 0; col <= width; col++)				// iterate through width
                        {
                                temp_a1 = 0, temp_a2 = 0, temp_a3 = 0;			// initialize temps to 0 
				temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;
                                index = (aisle * width * height) + (row * width) + col;	// calculate the index
                                local_col = col;					// calculate the ofset to not to write to halos in output
                                local_row = row + half;		
                                local_aisle = aisle;
                                local_width = width;
                                local_height = height;
                                local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;	// calculate the index to write to

                                for(int ref = 0; ref < filter_size; ref++)		// iterate the filter
                                {
                                        temp_a1 += T_x[index + (ref * width)].a1;	// sum up to temps along the y direction
                              		temp_a2 += T_x[index + (ref * width)].a2;
                              		temp_a3 += T_x[index + (ref * width)].a3;
                              		temp_b2 += T_x[index + (ref * width)].b2;
                                     	temp_b3 += T_x[index + (ref * width)].b3;
                                     	temp_c3 += T_x[index + (ref * width)].c3;
				}
				T_y[local_index].a1 = temp_a1 / filter_size;		// normalize and store the output to T_y
                                T_y[local_index].a2 = temp_a2 / filter_size;
                                T_y[local_index].a3 = temp_a3 / filter_size;
                                T_y[local_index].b2 = temp_b2 / filter_size;
                                T_y[local_index].b3 = temp_b3 / filter_size;
                                T_y[local_index].c3 = temp_c3 / filter_size;
                        }
                }
        }
	stop = chrono::high_resolution_clock::now();					// Record the stop event
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);		// calculate the time taken
	cout << "time taken to apply blur along y axis		: " << time.count()*1000 << " ms" << endl;	// Display the time taken

	T_x.clear();									// release T_x

// Apply the filter along Z axis
//
	vector<matrix> T_z(width * height * depth);					// Allocate vector for T_z
	start = chrono::high_resolution_clock::now();					// record start
        for(int aisle = 0; aisle <= depth - filter_size_z; aisle++)			// iterate the depth stop at point so filter could cover the rest
        {	
                for(int row = 0; row <= height; row++)					// iterate the height
                {
                        for(int col = 0; col <= width; col++)				// iterate the col
                        {	
                                temp_a1 = 0, temp_a2 = 0, temp_a3 = 0;			// initialize temps to zero
				temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;
                                index = (aisle * width * height) + (row * width) + col;	// calculate the index to read from
                                local_col = col;					// calcute the ofset to write into
                                local_row = row;
                                local_aisle = aisle + half_z;
                                local_width = width;
                                local_height = height;
                                local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;	// Calculate thhe index to write into

                                for(int ref = 0; ref < filter_size_z; ref++)		// iterate along the filter
                                {
                                	temp_a1 += T_y[index + (ref * width * height)].a1;	// accumulate to temp by readeing T_y along z direction
                                        temp_a2 += T_y[index + (ref * width * height)].a2;
                                        temp_a3 += T_y[index + (ref * width * height)].a3; 
                                        temp_b2 += T_y[index + (ref * width * height)].b2;
                                        temp_b3 += T_y[index + (ref * width * height)].b3;
                                        temp_c3 += T_y[index + (ref * width * height)].c3;
				}
				T_z[local_index].a1 = temp_a1 / filter_size_z;		// normalize and write to T_z
                                T_z[local_index].a2 = temp_a2 / filter_size_z;
                                T_z[local_index].a3 = temp_a3 / filter_size_z;		
                                T_z[local_index].b2 = temp_b2 / filter_size_z;
                                T_z[local_index].b3 = temp_b3 / filter_size_z;
                                T_z[local_index].c3 = temp_c3 / filter_size_z;
                        }
                }
        }
	stop = chrono::high_resolution_clock::now();					// record stop event
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);		// calculte the time taken
	cout << "time taken to apply blur along z axis		: " << time.count()*1000 << " ms" << endl;	// Display the time taken

	T_y.clear();									// Release T_y

// Remove the padding
//
	vector< Eigen::Matrix3f > T_m((width - 2 * half) * (height - 2 * half) * (depth - 2 * half_z));	// allocate the vector of Eigen Matrix type for future use
	start = chrono::high_resolution_clock::now();					// record start
	for(int aisle = half_z; aisle < depth - half_z; aisle++)			// iterate the depth between the data range
	{
		for(int row = half; row < height - half; row++)				// iterate the height between the data range
		{
			for(int col = half; col < width - half; col++)			// iterate the width between the data range
			{
				local_col = col - half;					// calculate the local indeces to write to by  ofsetting local variables
				local_row = row - half;
				local_aisle = aisle - half_z;
				local_width = width - 2 * half;
				local_height = height - 2 * half;
				local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;
				index = (aisle * width * height) + (row * width) + col;

				T_m[local_index] << T_z[index].a1, T_z[index].a2, T_z[index].a3, // Copy the data to Eigen matrix tyoe
						    T_z[index].a2, T_z[index].b2, T_z[index].b3, 
						    T_z[index].a3, T_z[index].b3, T_z[index].c3;
			}
		}
	}	
	stop = chrono::high_resolution_clock::now();					// record stop
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);		// calculate time taken
	cout << "time taken to remove padding 			: " << time.count()*1000 << " ms" << endl;	// Display the time taken

	T_z.clear();									// Release T_z

// Compute Eigen Vectors
//
	width = width - 2 * half;							// update width, height and depth after remove padding 
	height = height - 2 * half;
	depth = depth - 2 * half_z;

	eigen_vectors eigen_init = {0,0,0};
	vector<eigen_vectors>Evec(width * height * depth, eigen_init);			// create a vector of eigen vectors
	int nthreads;									// variable to store num of threads used
	double omp_start = omp_get_wtime();						// record start event
	#pragma omp parallel for							// openMP for loop parallelization
	for (unsigned int ii = 0; ii < Evec.size(); ii++)				// iterate the Tensors
	{
        	if(ii == 0)								// storing num of threads used only in first iteration
		{
            		nthreads=omp_get_num_threads();
        	}
		SelfAdjointEigenSolver<Matrix3f> handle;				// create a Eigen handle to solve the matrix 
		handle.computeDirect(T_m[ii]);						// compute the eigen vectors and store it in handle
		Eigen::Vector3f::Map(&Evec[ii].a) = handle.eigenvectors().col(0);	// chose only the first vector and map it to the std::vector type
	}
	double omp_stop = omp_get_wtime();						// record stop
	double omp_time = omp_stop - omp_start;						// calculate time taken

	cout << "time taken to calculate eigen vector		: " << omp_time << " s" << " (OpenMP " << nthreads<< " threads)" <<endl;	// Display the time taken

	T_m.clear();									// Free T_m

// Create a color Image									// This commented out section of code is used to create a colour image
/*											// which shows the direction represented using colour
	vector <float> 		l     		(width * height * depth, 0);		// The output nv can be imported to python and visulaized using imshow()
	vector <pixels>		nv    		(width * height * depth	  );
	vector <float> 	 	alpha 		(width * height * depth, 0);

	start = chrono::high_resolution_clock::now();
	for(int i = 0; i < l.size(); i++)
	{
		l[i] = sqrt( (pow(Evec[i].a,2) + pow(Evec[i].b,2) + pow(Evec[i].c,2) ));

		nv[i].r = Evec[i].a / l[i]; 
		nv[i].g = Evec[i].b / l[i]; 
		nv[i].b = Evec[i].c / l[i];
		
		alpha[i] = pixel[i] / 255.0f;
		
		nv[i].r = (abs(nv[i].r * alpha[i]));
		nv[i].g = (abs(nv[i].g * alpha[i]));
		nv[i].b = (abs(nv[i].b * alpha[i]));
	}
	stop = chrono::high_resolution_clock::now();
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to compute colour image		: " << time.count()*1000 << " ms" << endl;
	
	l.clear();
	alpha.clear();
	pixel.clear();
*/
// write a color image. I will import this file to python and visualize using matplotlib::imshow()
/*
	ofstream img_file("img.txt");
	for(auto &i : nv)
		img_file << i.r << " " << i.g << " " << i.b << " "; 
*/
// Apply Euler's method of integration

	vector< position > trace;							// Create a vector to store the trace. Size undefined at this point
	float del_t = 0.2;								// declare the timestep
	int n_steps = 1000000;								// number of steps to iterate

	omp_start = omp_get_wtime();							// record start 	
	#pragma omp parallel for private(index, trace) collapse(3)			// openmp parallelization
	for(int aisle = 0; aisle < 5; aisle++)						// iterate along depth
	{
		for(int row = 0; row < 5; row++)					// iterate along height
		{
			for(int col = 0; col < 5; col++)				// iterate along width
			{
				index = ((aisle) * height * width) + ((row) * height) + (col);	// Calculate the index. I will use this to write filename
				int tid=omp_get_thread_num();				// get my thread id
        			if(tid==0 && index < 1)
				{
            				nthreads=omp_get_num_threads();			// thread 0 will tell how many threads are used
        			}
				trace.clear();						// clear up trace for each iteration
				eulers_method(&trace, Evec, col, row, aisle, width, height, depth, del_t, n_steps);	// call the function to trace the path
				write_trace(trace, index);				// write the trace
			}
		}
	}	
	omp_stop = omp_get_wtime();							// record stop
	omp_time = (omp_stop - omp_start);						// calculate time taken
	cout << "time taken to do cryptography and writing it	: " << omp_time << " s" << " (OpenMP " << nthreads<< " threads)" << endl;	// Display the time taken

	trace.clear();									// clear the memory
	Evec.clear();
	pixel.clear();
	return 0;
}
