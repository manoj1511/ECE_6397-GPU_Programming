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

struct matrix
{
	float a1, a2, a3;
	float b1, b2, b3;
	float c1, c2, c3;
};


struct eigen_vectors
{	
	float a, b, c;
};

struct pixels
{
	float r, g, b;
};

struct position
{
	float x,y,z;
};

void eulers_method(vector<position> *trace, vector<eigen_vectors> Evec,int col, int row, int aisle, int width, int height, int depth, float del_t, int n)
{
	position initial = {(float) col, (float) row, (float) aisle};
	position new_pos = {0.0f, 0.0f, 0.0f};

	int init;

	(*trace).push_back(initial);	

	int step = 0;
	while(step < n)
	{	
		init   = ((int)initial.z * width * height) + ((int)initial.y * width) + ((int)initial.x);
		new_pos.x = initial.x + del_t * (Evec[init].a);
		new_pos.y = initial.y + del_t * (Evec[init].b);
		new_pos.z = initial.z + del_t * (Evec[init].c);

		if( (new_pos.x < width) && (new_pos.y < height) && (new_pos.z < depth) && (new_pos.x >= 0) && (new_pos.y >= 0) && (new_pos.z >= 0) && \
		    (initial.x < width) && (initial.y < height) && (initial.z < depth) && (initial.x >= 0) && (initial.y >= 0) && (initial.z >= 0) 	)
		{
			int new_p  = ((int)new_pos.z * width * height) + ((int)new_pos.y * width) + ((int)new_pos.x);

			float dot_p = (Evec[new_p].a * Evec[init].a) + (Evec[new_p].b * Evec[init].b) + (Evec[new_p].c * Evec[init].c);
	//		cout << "\n dot product of step: " << step << "is " << dot_p;
			if(dot_p < 0)
			{
				new_pos.x = initial.x + del_t * (-Evec[init].a);
				new_pos.y = initial.y + del_t * (-Evec[init].b);
				new_pos.z = initial.z + del_t * (-Evec[init].c);
			}

			(*trace).push_back(new_pos);
		//	cout << new_pos.x << " " << new_pos.y << " " << new_pos.z << endl;
			initial.x = new_pos.x;
			initial.y = new_pos.y;
			initial.z = new_pos.z;

			step++;

		}
		else break;
	}
	return;
}

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
	string pattern = "./kidney_100/test_*.jpg";
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

//

//	vector<string> filenames(2);
//	filenames[0] = "./kidney_100/test_001.jpg";
//	filenames[1] = "./kidney_100/test_002.jpg";
//
//	for(int it = 0; it<filenames.size();++it);
//		cout << filenames[it] << endl;
	cout << filenames.size()<<endl;
	cout << endl;
	int depth = filenames.size();

	vector<int> pixel;
	int width, height;

	chrono::high_resolution_clock::time_point start,stop;
	start = chrono::high_resolution_clock::now();

	for(auto &file : filenames)
	{	
		cimg_library::CImg<unsigned char> image(file.c_str());

		width = image.width();
		height = image.height();
//		cout << width<<" "<<height<<endl;	
		for( auto &it : image)
			pixel.push_back((int)it);
//		for( int i = 510; i < height*width; i+=width)
//			cout<<pixel[i]<<" ";
		
	}
//	cout << endl;
//	cout << "size of pixel : " <<pixel.size()<<endl;
//	cout << "size that has to be there : " << 511 * 511 * filenames.size()<<endl;


	stop = chrono::high_resolution_clock::now();
	
	chrono::duration<double> time;					
	
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to read files			: " << time.count()*1000 << " ms" << endl;

	float h = 1;
	float h2 = 4;
	unsigned int size = pixel.size();
//	cout << "files	: " << depth << endl;

	start = chrono::high_resolution_clock::now();

/************************ partial derivate along x direction *************************/

	vector<float> dx(size,0);

	for(unsigned int i = 0 ; i < size; i++)
	{
		if(i % width == 0) dx[i] = (pixel[i+1] - pixel[i])/h;
		else if(i % width == (unsigned int)(width-1)) dx[i] = (pixel[i] - pixel[i-1])/h;
		else dx[i] = (pixel[i+1] - pixel[i-1])/(2*h);
	}	

/*********************** partial derivate along y direction *************************/

	vector<float> dy(size,0);
	for(unsigned int i = 0; i < size; i++)
	{	
		int r = i/width;
		if(r % height == 0) dy[i] = (pixel[width+i] - pixel[i])/h;
		else if(r % height == height - 1) dy[i] = (pixel[i] - pixel[i-width])/h;
		else dy[i] = (pixel[i+width] - pixel[i-width])/(2*h);
	}

/*********************** partial derivate along z direction *************************/
	
	vector<float> dz(size,0);
	for(unsigned int i = 0; i < size; i++)
	{
		if(i < (unsigned int)(height*width)) dz[i] = (pixel[i + (width*height)] - pixel[i])/h2;
		else if(i >= (size - (height*width))) dz[i] = (pixel[i] - pixel[i - (width*height)])/h2;
		else dz[i] = (pixel[i + (width*height)] - pixel[i - (width*height)])/(2*h2);
	}


//	for(int i = (height*width); i < pixel.size()-((height-1)*width) ; i++)
//		cout << i-(width*height) << "	: " << dx[i] << endl;
//	for(int i = (height*width) ; i < pixel.size() ; i+=width)
//		cout << i-(height*width) << "	: " << dy[i] << endl;
//	for(int i = height*width-1 ; i < pixel.size() ; i+=(width*height))
//		cout << i << "	: " << dz[i] << endl;

	stop = chrono::high_resolution_clock::now();
	
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to calculate partial derivatives	: " << time.count()*1000 << " ms" << endl;
/*
	ofstream file("file5.txt");
	for(int i = 5*511*511; i < 6*511*511; i++)
	{
		file << pixel[i] << " ";
	}
	file.close();	
*/
/*
	ofstream file("part_der_x.txt");
	for(int i = 5*511*511; i < 6*511*511; i++)
	{
		file << dx[i] << " ";
	}
	file.close();	

	ofstream file_1("part_der_y.txt");
	for(int i = 5*511*511; i < 6*511*511; i++)
	{
		file_1 << dy[i] << " ";
	}
	file_1.close();

	ofstream file_2("part_der_z.txt");
	for(int i = 5*511*511; i < 6*511*511; i++)
	{
		file_2 << dz[i] << " ";
	}
	file_2.close();
*/	

	start = chrono::high_resolution_clock::now();

	matrix init = {0,0,0,0,0,0,0,0,0};
	vector<matrix> T(size,init);
//	cout << "Size of T vector	: " << T.size() << endl;
	for (unsigned int i = 0; i < size; i++)
	{
		T[i].a1 = dx[i] * dx[i];
		T[i].a2 = dx[i] * dy[i];
		T[i].a3 = dx[i] * dz[i];
		T[i].b2 = dy[i] * dy[i];
		T[i].b3 = dy[i] * dz[i];
		T[i].c3 = dz[i] * dz[i];
	}

	dx.clear();
	dy.clear();
	dz.clear();

	stop = chrono::high_resolution_clock::now();
	
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to calculate Tensor field		: " << time.count()*1000 << " ms" << endl;

/*	ofstream file_a1("tensor_field_a1.txt");
	for(int i = 0; i < T.size(); i++)
	{
		file_a1 << T[i].a1 << " ";
	}
	file_a1.close();	

	ofstream file_4("tensor_field_a2.txt");
	for(int i = 5*511*511; i < 6*511*511; i++)
	{
		file_4 << T[i].a2 << " ";
	}
	file_4.close();
	ofstream file_5("tensor_field_b2.txt");
	for(int i = 5*511*511; i < 6*511*511; i++)
	{
		file_5 << T[i].b2 << " ";
	}
	file_5.close();
	ofstream file_6("tensor_field_c3.txt");
	for(int i = 5*511*511; i < 6*511*511; i++)
	{
		file_6 << T[i].c3 << " ";
	}
	file_6.close();	
*/
/* 	
	int check = 0;
	cout << T[check].a1 << " " << T[check].a2 << " " << T[check].a3 << endl;
	cout << T[check].b1 << " " << T[check].b2 << " " << T[check].b3 << endl;
	cout << T[check].c1 << " " << T[check].c2 << " " << T[check].c3 << endl;
*/


// PADDING THE TENSOR FIELD FOR BLURRING
//
	int filter_size = 16;
        int half = filter_size / 2;


	int filter_size_z = 4;
        int half_z = filter_size_z / 2;


	width = width + 2 * half;
        height = height + 2 * half;
        depth = depth + 2 * half_z;
        vector<matrix> T_pad(width * height * depth);


	int index;
        int local_row, local_col, local_aisle, local_width, local_height, local_index;
	
	start = chrono::high_resolution_clock::now();
	for(int aisle = 0; aisle < depth; aisle++)
        {
                for(int row = 0; row < height; row++)
                {
                        for(int col = 0; col < width; col++)
                        {
                                index = (aisle * width * height) + (row * width) + col;
                                local_col = col - half;
                                local_row = row - half;
                                local_aisle = aisle - half_z;
                                local_width = width - 2 * half;
                                local_height = height - 2 * half;
                                local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;

                                if(col >= half && col < (width - half) && row >= half && row < (height - half) && aisle >= half_z && aisle < (depth - half_z))
                                {
                                        T_pad[index].a1 = T[local_index].a1;
                                        T_pad[index].a2 = T[local_index].a2;
                                        T_pad[index].a3 = T[local_index].a3;
                                        T_pad[index].b2 = T[local_index].b2;
                                        T_pad[index].b3 = T[local_index].b3;
                                        T_pad[index].c3 = T[local_index].c3;
                                }
				else
				{
                                        T_pad[index].a1 = 0;
                                        T_pad[index].a2 = 0;
                                        T_pad[index].a3 = 0;
                                        T_pad[index].b2 = 0;
                                        T_pad[index].b3 = 0;
                                        T_pad[index].c3 = 0;
				}
                        }
                }
        }
	stop = chrono::high_resolution_clock::now();
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to apply Padding			: " << time.count()*1000 << " ms" << endl;

	ofstream T_pad_w("T_pad_cpu", ios::binary);
	T_pad_w.write((char *) &T_pad[0].a1, T_pad.size() * sizeof(matrix));	
	T_pad_w.clear();

	T.clear();

// Apply the filter along X axis
	
	vector<matrix> T_x(width * height * depth);
	float temp_a1, temp_a2, temp_a3, temp_b2, temp_b3, temp_c3;

	start = chrono::high_resolution_clock::now();
	for(int aisle = 0; aisle < depth; aisle++)
        {
                for(int row = 0; row < height; row++)
                {
                        for(int col = 0; col <= width - filter_size; col++)
                        {
                                temp_a1 = 0, temp_a2 = 0, temp_a3 = 0, temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;
                                index = (aisle * width * height) + (row * width) + col;
                                local_col = col + half;
                                local_row = row;
                                local_aisle = aisle;
                                local_width = width;
                                local_height = height;
                                local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;

                                for(int ref = 0; ref < filter_size; ref++)
                                {
                                        temp_a1 += T_pad[index + ref].a1;
                                        temp_a2 += T_pad[index + ref].a2;
                                        temp_a3 += T_pad[index + ref].a3;
                                        temp_b2 += T_pad[index + ref].b2;
                                        temp_b3 += T_pad[index + ref].b3;
                                        temp_c3 += T_pad[index + ref].c3;
                                }
                                T_x[local_index].a1 = temp_a1 / filter_size;
                                T_x[local_index].a2 = temp_a2 / filter_size;
                                T_x[local_index].a3 = temp_a3 / filter_size;
                                T_x[local_index].b2 = temp_b2 / filter_size;
                                T_x[local_index].b3 = temp_b3 / filter_size;
                                T_x[local_index].c3 = temp_c3 / filter_size;
                        }
                }
        }
	stop = chrono::high_resolution_clock::now();
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to apply blur along x axis		: " << time.count()*1000 << " ms" << endl;
/*
	ofstream file_T_x("T_x.txt");
	for(int i = 0; i < T_x.size(); i++)
	{
		file_T_x << T_x[i].a1 << " ";
	}
	file_T_x.close();
*/
	T_pad.clear();

// Apply the filter along Y axis

	vector<matrix> T_y(width * height * depth);
	start = chrono::high_resolution_clock::now();
	for(int aisle = 0; aisle < depth; aisle++)
        {
                for(int row = 0; row <= height - filter_size; row++)
                {
                        for(int col = 0; col <= width; col++)
                        {
                                temp_a1 = 0, temp_a2 = 0, temp_a3 = 0, temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;
                                index = (aisle * width * height) + (row * width) + col;
                                local_col = col;
                                local_row = row + half;
                                local_aisle = aisle;
                                local_width = width;
                                local_height = height;
                                local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;

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
                }
        }
	stop = chrono::high_resolution_clock::now();
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to apply blur along y axis		: " << time.count()*1000 << " ms" << endl;
/*
	ofstream file_T_y("T_y.txt");
	for(int i = 0; i < T_y.size(); i++)
	{
		file_T_y << T_y[i].a1 << " ";
	}
	file_T_y.close();
*/
	T_x.clear();

// Apply the filter along Z axis

	vector<matrix> T_z(width * height * depth);

	start = chrono::high_resolution_clock::now();
        for(int aisle = 0; aisle <= depth - filter_size_z; aisle++)
        {
                for(int row = 0; row <= height; row++)
                {
                        for(int col = 0; col <= width; col++)
                        {	
                                temp_a1 = 0, temp_a2 = 0, temp_a3 = 0, temp_b2 = 0, temp_b3 = 0, temp_c3 = 0;
                                index = (aisle * width * height) + (row * width) + col;
                                local_col = col;
                                local_row = row;
                                local_aisle = aisle + half_z;
                                local_width = width;
                                local_height = height;
                                local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;

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
                }
        }
	stop = chrono::high_resolution_clock::now();
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to apply blur along z axis		: " << time.count()*1000 << " ms" << endl;
/*
	ofstream file_T_z("T_z.txt");
	for(int i = 0; i < T_z.size(); i++)
	{
		file_T_z << T_z[i].a1 << " ";
	}
	file_T_z.close();
*/
	T_y.clear();

// Remove the padding

	vector< Eigen::Matrix3f > T_m((width - 2 * half) * (height - 2 * half) * (depth - 2 * half_z));

	start = chrono::high_resolution_clock::now();
	for(int aisle = half_z; aisle < depth - half_z; aisle++)
	{
		for(int row = half; row < height - half; row++)
		{
			for(int col = half; col < width - half; col++)
			{
				local_col = col - half;
				local_row = row - half;
				local_aisle = aisle - half_z;
				local_width = width - 2 * half;
				local_height = height - 2 * half;
				local_index = (local_aisle * local_width * local_height) + (local_row * local_width) + local_col;
				index = (aisle * width * height) + (row * width) + col;

				T_m[local_index] << T_z[index].a1, T_z[index].a2, T_z[index].a3, 
						    T_z[index].a2, T_z[index].b2, T_z[index].b3, 
						    T_z[index].a3, T_z[index].b3, T_z[index].c3;
			}
		}
	}	
	stop = chrono::high_resolution_clock::now();
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to remove padding 			: " << time.count()*1000 << " ms" << endl;
	T_z.clear();

// Compute Eigen Vectors

	width = width - 2 * half;
	height = height - 2 * half;
	depth = depth - 2 * half_z;

	eigen_vectors eigen_init = {0,0,0};
	vector<eigen_vectors>Evec(width * height * depth, eigen_init);


	double omp_start = omp_get_wtime();
	#pragma omp parallel for
	for (unsigned int ii = 0; ii < Evec.size(); ii++)	
	{
		SelfAdjointEigenSolver<Matrix3f> handle;
		handle.computeDirect(T_m[ii]);
		Eigen::Vector3f::Map(&Evec[ii].a) = handle.eigenvectors().col(0);
	}
	double omp_stop = omp_get_wtime();
	double omp_time = omp_stop - omp_start;

	cout << "time taken to calculate eigen vector		: " << omp_time << " s" << endl;

/*
	ofstream Evec_file("Evec_vectors");
	char *ptr = (char *)&Evec[0].a;
	Evec_file.write(ptr, 3 * width * height * depth * sizeof(float));
	Evec_file.close();

	ofstream vector("vectors.txt");
	
	for(int i = 0; i < Evec.size(); i++)
	{
		vector << Evec[i].a << " " << Evec[i].b << " " << Evec[i].c << " ";
	}
	vector.close();

	cout <<  "T_m[0] is :" << endl;
	cout << T_m[0] << endl;
	cout << "-------------------" << endl << endl; 
	
	handle.computeDirect(T_m[0]);
	cout << handle.eigenvalues() << endl<< endl;
	cout << handle.eigenvectors() << endl;
*/
	T_m.clear();

// Create a color Image
/*
	vector <float> 		l     		(width * height * depth, 0);
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

	vector< position > trace;

	float del_t = 0.2;	
	int n_steps = 100000;
	
	omp_start = omp_get_wtime();

	#pragma omp parallel for private(index, trace) collapse(3)
	for(int aisle = 0; aisle < 5; aisle++)
	{
		for(int row = 0; row < 5; row++)
		{
			for(int col = 0; col < 5; col++)
			{
				index = ((aisle) * height * width) + ((row) * height) + (col);
				int tid=omp_get_thread_num();
        			if(tid==0 && index < 1)
				{
            				int nthreads=omp_get_num_threads();
            				cout << "Number of threads = " << nthreads << endl;
        			}
				trace.clear();
				eulers_method(&trace, Evec, col, row, aisle, width, height, depth, del_t, n_steps);
				write_trace(trace, index);
			}
		}
	}	
	omp_stop = omp_get_wtime();
	omp_time = (omp_stop - omp_start);

	cout << "time taken to do cryptography and writing it	: " << omp_time << " s" << endl;

//	nv.clear();
	trace.clear();

	ofstream Evec_w("Evec.bin", ios::binary);
	Evec_w.write((char *) &Evec[0].a, size * 3 * sizeof(float));
	Evec_w.close();

	return 0;
}
