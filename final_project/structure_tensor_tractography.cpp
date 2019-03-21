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

#define pi 3.141
#define mu 0


using namespace std;
using namespace cimg_library;

struct matrix
{
	float a1, a2, a3;
	float b1, b2, b3;
	float c1, c2, c3;
};

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
	cout << endl;
	cout << "size of pixel : " <<pixel.size()<<endl;
	cout << "size that has to be there : " << 511 * 511 * filenames.size()<<endl;

	float h = 1;
	float h2 = 4;
	unsigned int size = pixel.size();
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


	pixel.clear();

/************************ Tensor field calculation *************************/

	matrix init = {0,0,0,0,0,0,0,0,0};
	vector<matrix> T(size,init);
//	cout << "Size of T vector	: " << T.size() << endl;
	for (int i = 0; i < size; i++)
	{
		T[i].a1 = dx[i] * dx[i];
		T[i].a2 = dx[i] * dy[i];
		T[i].a3 = dx[i] * dz[i];
		T[i].b1 = dy[i] * dx[i];
		T[i].b2 = dy[i] * dy[i];
		T[i].b3 = dy[i] * dz[i];
		T[i].c1 = dz[i] * dx[i];
		T[i].c2 = dz[i] * dy[i];
		T[i].c3 = dz[i] * dz[i];
	} 	
	int check = 0;
	cout << T[check].a1 << " " << T[check].a2 << " " << T[check].a3 << endl;
	cout << T[check].b1 << " " << T[check].b2 << " " << T[check].b3 << endl;
	cout << T[check].c1 << " " << T[check].c2 << " " << T[check].c3 << endl;


/************************ Blur the tensors *************************/
// I am using gaussian kernel to blur

  	double coeff;
	int sigma = 32;									
	coeff = 1/sqrt(2*sigma*sigma*pi);				
	cout << "coefficient is : " << coeff << endl << endl;	

	int k = 6 * sigma;
	if(k%2==0) k++;	
	int k_half = k/2;

	vector<float> K(k,0);

	float sum = 0;		
	for(int i=-k_half; i<=k_half; i++)		
	{
		K[i+k_half]=coeff*exp(-(((i-mu)*(i-mu))/(2*sigma*sigma)));	
		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;
		sum += K[i+k_half];			
	}
	
	cout << "Sum is 	: " << sum << endl;
	cout << "-----------------------" << endl;  
	cout << "Normalized K" << endl;  
	cout << "-----------------------" << endl;  

	float sum2 = 0;
	for (int i=-k_half; i<=k_half; i++)
	{
		K[i+k_half]/=sum;
		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;
		sum2+=K[i+k_half];				
	}


	cout << "Sum is 	: " << sum2 << endl;

/************************ Gaussian along X axis *************************/

	float temp1 = 0;
	vector<matrix> T_x(size,init);

	for(int k = 0; k < depth; k++)
	{
		for(int j = 0; j < height; j++)
		{
			for(int i = 0; i < width-k; i++)
			{
				int index = (k*width*height) + (j*width) + i;
				for(int ref = 0; ref < k; ref++)
				{
					temp1 += K[ref] + T[index+ref].a1;
				}
				T_x[index].a1;
			}
		}
	}

/************************ Gaussian along Y axis *************************/
/*
	vector<matrix> T_y(size,init);

	for(int k = 0; k < depth; k++)
	{
		for(int j = 0; j < height-k; j++)
		{
			for(int i = 0; j < width-k; j++)
			{
				int index = (k*width*height) + (j*width) + i;
				for(int ref = 0; ref < k; ref++)
				{
					temp1 += K[ref] + T_x[index+ref].a1;
				}
				T_y[index].a1;
			}
		}
	}
*/

	

	T_x.clear();
	dx.clear();
	dy.clear();
	dz.clear();
	return 0;
}
