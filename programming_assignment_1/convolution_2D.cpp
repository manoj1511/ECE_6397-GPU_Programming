#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace std;

struct Pixel {unsigned char r, g, b; };

int main(int argc, char* argv[])
{
	if(argc !=3)
	{
		cout<<"Please enter 3 arguments";
		cout<<"USAGE:./executable image_file_name.ppm kernal_size ";
		return 1;
	}

	string filename = argv[1];
	int sigma = atoi(argv[2]);

	ifstream file(filename.c_str(), ios::binary);
	string type;
	int height,width,range;

	file >> type >> height >> width >> range;

	int N = height*width;	

 	size_t pixel_size = height * width * sizeof(Pixel);	
	
	cout<< "size to be allocated 	: " << pixel_size << endl;
	cout << type << endl << height << endl << width << endl << range << endl; 

	Pixel *pixel = new Pixel[N];	
	
	for(int i=0; i<N; i++)
	{
		file >> pixel[i].r; cout << (int)pixel[i].r << " ";
		file >> pixel[i].g; cout << (int)pixel[i].g << " ";
		file >> pixel[i].b; cout << (int)pixel[i].b << endl;
	}

	file.close();
	return 0;
}
