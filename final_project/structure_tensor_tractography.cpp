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

using namespace std;
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

	float h=1;

/************************ partial derivate along x direction *************************/

	vector<float> dx(pixel.size(),0);
	for(int i = 0 ; i < pixel.size(); i++)
	{
		if(i % width == 0) dx[i] = (pixel[i+1] - pixel[i])/h;
		else if(i % width == width-1) dx[i] = (pixel[i] - pixel[i-1])/h;
		else dx[i] = (pixel[i+1] - pixel[i-1])/(2*h);
	}	

/************************ partial derivate along y direction *************************/

	vector<float> dy(pixel.size(),0);
	int row;
	for(int i = 0; i < pixel.size(); i++)
	{	
		row = i/width;
		if(row % height == 0) dy[i] = (pixel[width+i] - pixel[i])/h;
		else if(row % height == height - 1) dy[i] = (pixel[i] - pixel[i-width])/h;
		else dy[i] = (pixel[i+width] - pixel[i-width])/(2*h);
	}

/************************ partial derivate along z direction *************************/
	
	float h2=4;
	vector<float> dz(pixel.size(),0);
	for(int i = 0; i < pixel.size(); i++)
	{
		if(i < height*width) dz[i] = (pixel[i + (width*height)] - pixel[i])/h2;
		else if(i >= (pixel.size() - (height*width))) dz[i] = (pixel[i] - pixel[i - (width*height)])/h2;
		else dz[i] = (pixel[i + (width*height)] - pixel[i - (width*height)])/(2*h2);
	}


//	for(int i = (height*width); i < pixel.size()-((height-1)*width) ; i++)
//		cout << i-(width*height) << "	: " << dx[i] << endl;
//	for(int i = (height*width) ; i < pixel.size() ; i+=width)
//		cout << i-(height*width) << "	: " << dy[i] << endl;
	for(int i = height*width-1 ; i < pixel.size() ; i+=(width*height))
		cout << i << "	: " << dz[i] << endl;




/*
	dx[0] = (pixel[1] - pixel[0])/h;
	for(int i = 1 ; i < width - 1 ; i++)
		dx[i] = (pixel[i+1] - pixel[i-1])/(2*h);
	dx[510] = (pixel[width-1]-pixel[width-2])/h;
	for(int i = 0 ; i < width ; i++)
		cout << i<<"	: "<< dx[i] << endl;

	vector<float> dy(pixel.size(),0);
	dy[0] = (pixel[width] - pixel[0]);
	for(int i = width ; i < pixel.size(); i++)
		dy[i]
*/	
	pixel.clear();
	dx.clear();
	dy.clear();
	return 0;
}
