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
	string pattern = "./kidney_100/*.jpg";
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

//	for(int it = 0; it<filenames.size();++it);
//		cout << filenames[it] << endl;
	cout << filenames.size()<<endl;
	cout << endl;
	globfree(&glob_result);

	vector<int> pixel;

	for(int i=0; i < filenames.size(); i++)
	{	
		cimg_library::CImg<unsigned char> image(filenames[i].c_str());

		int width = image.width();
		int height = image.height();
//		cout << width<<" "<<height<<endl;	
		for( auto &it : image)
			pixel.push_back((int)it);
	}
	cout << "size of pixel : " <<pixel.size()<<endl;
	cout << "size that has to be there : " << 511 * 511 * filenames.size()<<endl;

	
	pixel.clear();
	return 0;
}
