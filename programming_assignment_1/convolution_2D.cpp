#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace std;

struct Pixel {float r, g, b; };

int main(int argc, char* argv[])
{
	if(argc !=3)									//Checking if there are 3 arguments
	{
		cout<<"Please enter 3 arguments";					//returns this message if 3 arguments not present
		cout<<"USAGE:./executable image_file_name.ppm kernal_size ";		
		return 1;
	}

	string filename = argv[1];
	int sigma = atoi(argv[2]);

	ifstream file(filename.c_str(), ios::binary);

	string type;

	string comment;
	string dustbin;

	file >> type; 
	file >> comment;								// comment variable to store comments


	while(comment.compare(0,1,"#") == 0)
	{
		getline(file,dustbin);
		file >> comment;
	}

	int height,width,range;

	height = stoi(comment);
	file >> width >> range;

	int N = height*width;	

 	size_t pixel_size = height * width * sizeof(Pixel);	
	
	cout<< "size to be allocated 	: " << pixel_size << endl;
	cout << type << endl << height << endl << width << endl << range << endl; 	

	Pixel *pixel = new Pixel[N];							// allocate array to store pixel values

	unsigned char temp;								// temp reg to store pixel values
	
	for(int i=0; i<N; i++)
	{
		file >> temp; pixel[i].r = (float)temp;
		file >> temp; pixel[i].g = (float)temp; 
		file >> temp; pixel[i].b = (float)temp; 
	}

	delete[] pixel;
	file.close();
	return 0;
}
