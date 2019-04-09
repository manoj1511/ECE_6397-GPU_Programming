// Author : Manoj Kumar Cebol Sundarrajan
// Organization : University of Houston

#include<iostream>
#include<vector>
#include<glob.h>
#include<string>
#include<cstring>
#include<sstream>
#include<stdexcept>
#include"CImg.h"
#ifdef Success
  #undef Success
#endif
#include<chrono>
//#include<mkl_lapacke.h>
//#define EIGEN_USE_MKL_ALL
#include<Eigen>
#include<cmath>

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

struct pixel
{
	float r, g, b;
};

vector<float> create_kernel(const int s, int *k_ele)
{

  	double coeff;
	coeff = 1/sqrt(2*s*s*pi);				
//	cout << "coefficient is : " << coeff << endl << endl;	

	*k_ele = 6 * s;
	if(*k_ele%2==0) (*k_ele)++;
	int k = *k_ele;	
	int k_half = k/2;

	vector<float> K(k,0);

	float sum = 0;		
	for(int i=-k_half; i<=k_half; i++)		
	{
		K[i+k_half]=coeff*exp(-(((i-mu)*(i-mu))/(2*s*s)));	
//		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;
		sum += K[i+k_half];			
	}
	
//	cout << "Sum is 	: " << sum << endl;
//	cout << "-----------------------" << endl;  
//	cout << "Normalized K" << endl;  
//	cout << "-----------------------" << endl;  

	float sum2 = 0;
	for (int i=-k_half; i<=k_half; i++)
	{
		K[i+k_half]/=sum;
//		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;
		sum2+=K[i+k_half];				
	}
//	cout << "Sum is 	: " << sum2 << endl;

	return K;

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

	for(int i = 0; i < glob_result.gl_pathc; i++)
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


	start = chrono::high_resolution_clock::now();

/************************ partial derivate along x direction *************************/

	vector<float> dx(size,0);

	for(int i = 0 ; i < size; i++)
	{
		if(i % width == 0) dx[i] = (pixel[i+1] - pixel[i])/h;
		else if(i % width == width-1) dx[i] = (pixel[i] - pixel[i-1])/h;
		else dx[i] = (pixel[i+1] - pixel[i-1])/(2*h);
	}	

/************************ partial derivate along y direction *************************/

	vector<float> dy(size,0);
	int row;
	for(int i = 0; i < size; i++)
	{	
		row = i/width;
		if(row % height == 0) dy[i] = (pixel[width+i] - pixel[i])/h;
		else if(row % height == height - 1) dy[i] = (pixel[i] - pixel[i-width])/h;
		else dy[i] = (pixel[i+width] - pixel[i-width])/(2*h);
	}

/************************ partial derivate along z direction *************************/
	
	vector<float> dz(size,0);
	for(int i = 0; i < size; i++)
	{
		if(i < height*width) dz[i] = (pixel[i + (width*height)] - pixel[i])/h2;
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

	pixel.clear();

/************************ Tensor field calculation *************************/

	start = chrono::high_resolution_clock::now();

	matrix init = {0,0,0,0,0,0,0,0,0};
	vector<matrix> T(size,init);
//	cout << "Size of T vector	: " << T.size() << endl;
	for (int i = 0; i < size; i++)
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
	
/* 	
	int check = 0;
	cout << T[check].a1 << " " << T[check].a2 << " " << T[check].a3 << endl;
	cout << T[check].b1 << " " << T[check].b2 << " " << T[check].b3 << endl;
	cout << T[check].c1 << " " << T[check].c2 << " " << T[check].c3 << endl;
*/

	vector<float> K;

	int k_ele = 0;

	K = create_kernel(1, &k_ele);

/************************************************************************/
/*
	matrix init2 = {255,0,0,0,0,0,0,0,0};
	vector<matrix> T_new(8000, init2);
	T.clear();
	T = T_new;
	vector<matrix> T_x(8000,init);
	vector<matrix> T_y(8000,init);
	vector<matrix> T_z(8000,init);
	height = 20; 
	depth = 20;
	width = 20;
*/
/************************************************************************/
//print T
/************************************************************************/
/*
	cout << "Printing T" << endl;
	for(int k = 0; k < 2; k++)
	{
		for(int j = 0; j < 20; j++)
		{	
			for(int i = 0; i < 20; i++)
			{
				int index = (k*width*height) + (j*width) + i;
				cout << T[index].a1 << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << endl;
*/
/************************************************************************/


/************************ Gaussian along X axis *************************/

	float temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

	start = chrono::high_resolution_clock::now();

	vector<matrix> T_x(size,init);

	for(int k = 0; k < depth; k++)
	{
		for(int j = 0; j < height; j++)
		{
			for(int i = 0; i <= width-k_ele; i++)
			{

				temp1 = 0, temp2 = 0, temp3 = 0;
				temp4 = 0, temp5 = 0, temp6 = 0;
				temp7 = 0, temp8 = 0, temp9 = 0;

				int index = (k*width*height) + (j*width) + i;
				for(int ref = 0; ref < k_ele; ref++)
				{
					temp1 += K[ref] * T[index+ref].a1;
					temp2 += K[ref] * T[index+ref].a2;
					temp3 += K[ref] * T[index+ref].a3;
					temp5 += K[ref] * T[index+ref].b2;
					temp6 += K[ref] * T[index+ref].b3;
					temp9 += K[ref] * T[index+ref].c3;
				}
				T_x[index].a1 = temp1;
				T_x[index].a2 = temp2;
				T_x[index].a3 = temp3;
				T_x[index].b1 = T_x[index].a2;
				T_x[index].b2 = temp5;
				T_x[index].b3 = temp6;
				T_x[index].c1 = T_x[index].a3;
				T_x[index].c2 = T_x[index].b3;
				T_x[index].c3 = temp9;
			}
		}
	}

	stop = chrono::high_resolution_clock::now();

	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to apply blur along X axis		: " << time.count()* 1000 << " ms" << endl;
	

/************************************************************************/
/*
	cout << "Printing T_x" << endl;
	for(int k = 0; k < 2; k++)
	{
		for(int j = 0; j < 20; j++)
		{	
			for(int i = 0; i < 20; i++)
			{
				int index = (k*width*height) + (j*width) + i;
				cout << T_x[index].a1 << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << endl;
*/
/************************************************************************/

/************************ Gaussian along Y axis *************************/

	start = chrono::high_resolution_clock::now();
	
	vector<matrix> T_y(size,init);

	for(int k = 0; k < depth; k++)
	{
		for(int j = 0; j <= height-k_ele; j++)
		{
			for(int i = 0; i <= width-k_ele; i++)
			{
	
				temp1 = 0, temp2 = 0, temp3 = 0;
				temp4 = 0, temp5 = 0, temp6 = 0;
				temp7 = 0, temp8 = 0, temp9 = 0;

				int index = (k*width*height) + (j*width) + i;
				for(int ref = 0; ref < k_ele; ref++)
				{
					temp1 += K[ref] * T_x[index+(ref*width)].a1;
					temp2 += K[ref] * T_x[index+(ref*width)].a2;
					temp3 += K[ref] * T_x[index+(ref*width)].a3;
					temp5 += K[ref] * T_x[index+(ref*width)].b2;
					temp6 += K[ref] * T_x[index+(ref*width)].b3;
					temp9 += K[ref] * T_x[index+(ref*width)].c3;
				}
				T_y[index].a1 = temp1;
				T_y[index].a2 = temp2;
				T_y[index].a3 = temp3;
				T_y[index].b1 = T_y[index].a2; 
				T_y[index].b2 = temp5;
				T_y[index].b3 = temp6;
				T_y[index].c1 = T_y[index].a3;  
				T_y[index].c2 = T_y[index].b3; 
				T_y[index].c3 = temp9;
			}
		}
	}

	stop = chrono::high_resolution_clock::now();

	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to apply blur along Y axis		: " << time.count()*1000 << " ms" << endl;
	
	T_x.clear(); 					// I dont need T_x anymore

/************************************************************************/
/*
	cout << "Printing T_y" << endl;
	for(int k = 0; k < 2; k++)
	{
		for(int j = 0; j < 20; j++)
		{	
			for(int i = 0; i < 20; i++)
			{
				int index = (k*width*height) + (j*width) + i;
				cout << T_y[index].a1 << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << endl;
*/
/************************************************************************/


/************************* Gaussian along Z axis *************************/

	K.clear();
	K = create_kernel(1, &k_ele);	
	
//	for(auto &i : K)
//		cout << i << " ";
//	cout << endl << endl;	

	start = chrono::high_resolution_clock::now();

	int new_width  = width  - k_ele;
	int new_height = height - k_ele;
	int new_depth  = depth  - k_ele;
	int new_size   = new_width * new_height * new_depth;

	vector< Eigen::Matrix3f > T_m(new_size);	

	for(int k = 0; k < new_depth; k++)
	{
		for(int j = 0; j < new_height; j++)
		{
			for(int i = 0; i < new_width; i++)
			{
				temp1 = 0, temp2 = 0, temp3 = 0;
				temp4 = 0, temp5 = 0, temp6 = 0;
				temp7 = 0, temp8 = 0, temp9 = 0;

				int index = (k*width*height) + (j*width) + i;
				int index_2 = (k*new_width*new_height) + (j*new_width) + i;
				for(int ref = 0; ref < k_ele; ref++)
				{
					temp1 += K[ref] * T_y[index+(ref*width*height)].a1;
					temp2 += K[ref] * T_y[index+(ref*width*height)].a2;
					temp3 += K[ref] * T_y[index+(ref*width*height)].a3;
					temp5 += K[ref] * T_y[index+(ref*width*height)].b2;
					temp6 += K[ref] * T_y[index+(ref*width*height)].b3;
					temp9 += K[ref] * T_y[index+(ref*width*height)].c3;
				}
	
				T_m[index_2] << temp1, temp2, temp3, temp2, temp5, temp6, temp3, temp6, temp9;
			}
		}
	}


	stop = chrono::high_resolution_clock::now();

	time = chrono::duration_cast< chrono::duration<double> >(stop - start);

	cout << "time taken to apply blur along Z axis		: " << time.count() * 1000<< " ms" << endl;
	
	T_y.clear();					// I dont need T_y anymore

/************************** Calc Evec *********************************/

	eigen_vectors eigen_init = {0,0,0};

	vector<eigen_vectors>Evec(new_size, eigen_init);

	cout << "Calculating Eigen "<<endl;	

	start = chrono::high_resolution_clock::now();

	SelfAdjointEigenSolver<Matrix3f> handle;

	for (int ii = 0; ii < new_size; ii++)	
	{
		handle.computeDirect(T_m[ii]);
		Eigen::Vector3f::Map(&Evec[ii].a) = handle.eigenvectors().col(0);
	}

	stop = chrono::high_resolution_clock::now();
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);
	cout << "time taken to calcuate Eigen			: " << time.count() * 1000 << " ms" << endl;
//	cout << Evec[0].a <<" "<<Evec[0].b <<" "<<Evec[0].c;
		
/************************************************************************/

	vector<float> vec_length(new_size, 0);
	vector<eigen_vectors> norm_Evec(new_size, eigen_init);

	float temp1a = 0.0f;
	float temp2a = 0.0f;
	float temp3a = 0.0f;
	start = chrono::high_resolution_clock::now();
	for(int i = 0; i < new_size; i++)
	{
		temp1a = pow(Evec[i].a,2);
		temp2a = pow(Evec[i].b,2);
		temp3a = pow(Evec[i].c,2);
		vec_length[i] = sqrt(temp1+temp2+temp3);
	}
	for(int i = 0; i < new_size; i++)
	{
		norm_Evec[i].a = Evec[i].a / vec_length[i];
		norm_Evec[i].b = Evec[i].b / vec_length[i];
		norm_Evec[i].c = Evec[i].c / vec_length[i];
	}
	stop = chrono::high_resolution_clock::now();
	time = chrono::duration_cast< chrono::duration<double> >(stop - start);
	cout << "time taken for post calculation after Eigen	: " << time.count() * 1000 << " ms" << endl;
	
	
	return 0;
}
