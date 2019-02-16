#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>

#define mu 0
#define pi 3.141

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
	double coeff;

/***************************READ FILE*****************************************/

	ifstream file(filename.c_str(), ios::binary);

	string type;
	string comment;
	string dustbin;									// unused dustin variable to save comments

	file >> type; 
	file >> comment;								// comment variable to store comments


	while(comment.compare(0,1,"#") == 0)						// see if first character is a #
	{
		getline(file,dustbin);							// then read the entire line
		file >> comment;							// read the next word
	}
	
	int height,width,range;

	height = stoi(comment);								// value after comment store it in height
	file >> width >> range;								// continue reading for width and range

	int N = height*width;	
 	size_t pixel_size = height * width * sizeof(Pixel);				// calulate the size needed to allocate
	
	cout<< "size to be allocated 	: " << pixel_size << endl;
	cout << type << endl << height << endl << width << endl << range << endl; 	// display the headers

	Pixel *pixel = new Pixel[N];							// allocate array to store pixel values

	unsigned char temp;								// temp reg to store pixel values
	
	for(int i=0; i<N; i++)
	{
		file >> temp; pixel[i].r = (float)temp;					// read pixel values from file one by one and convert into float
		file >> temp; pixel[i].g = (float)temp; 
		file >> temp; pixel[i].b = (float)temp;
	}

	file.close();

/*****************************CREATE KERNEL***********************************/
	
	coeff = 1/sqrt(2*sigma*sigma*pi);
	cout << "coefficient is :" << coeff<<endl;
	
	int k = 6 * sigma;
	if(k%2==0) k++;

	int k_half = k/2;
	
	float* K = (float*) malloc(k * sizeof(float));

	for(int i=-k_half; i<=k_half; i++)
	{
		K[i]=coeff*exp(-(((i-mu)*(i-mu))/(2*sigma*sigma)));
		cout << "k["<<i<<"]	:" << K[i]<<endl;
	}

/**************************CONVOLUTION ROW WISE*******************************/

float temp1, temp2, temp3;

	for(int i=k_half; i<=width-k_half+1; i++)							// stops at width so that ref can fill rest of image width
	{
		temp1 = 0, temp2 = 0, temp3 = 0;
		for(int j=0; j<height; j++)
		{
			for(int ref=-k_half; ref<=k_half; ref++ )
			{
				temp1 += K[ref] * pixel[(i+j*width)-ref].r;
				temp2 += K[ref] * pixel[(i+j*width)-ref].g;
				temp3 += K[ref] * pixel[(i+j*width)-ref].b;
			}
			
			pixel[i+j*width].r = temp1;		   
			pixel[i+j*width].g = temp2;		   
			pixel[i+j*width].b = temp3;		   
		}
	}
//	for(int i=0; i<N; i++)
//	{
//		if(i%5 == 0) cout << endl;
//		cout << pixel[i].b << "		";
//
//		
//	}
//	cout << endl;

/***********************CONVOLUTION COLUMN WISE******************************/



	for(int i=0; i<width; i++)		
	{
		temp1 = 0; temp2 = 0 ; temp3 = 0;
		for(int j=k_half; j<=height-k_half+1; j++)						// stops at a hight so that ref can fill rest of image height
		{
			for(int ref=-k_half; ref<=k_half; ref++ )
			{	
				temp1 += K[ref] * pixel[(j+i*width)-ref].r;
				temp2 += K[ref] * pixel[(j+i*width)-ref].g;
				temp3 += K[ref] * pixel[(j+i*width)-ref].b;
			}
			
			pixel[j+i*width].r = temp1;		   
			pixel[j+i*width].g = temp2;		   
			pixel[j+i*width].b = temp3;		   
		}
	}
	cout <<"\n----------------------------------------------------------\n";
	for(int i=0; i<N; i++)
	{
		if(i%5 == 0) cout << endl;
		cout << pixel[i].b << "		";

		
	}
	cout << endl;

/******************************WRITE FILE*************************************/

	ofstream wfile("output_image.ppm", ios::binary);
 	wfile << type << endl;
 	wfile << height << " " << width << endl  << range << endl;

	for (int i = 0; i < N; i++) 
	{
		wfile << (unsigned char)pixel[i].r;
		wfile << (unsigned char)pixel[i].g;
		wfile << (unsigned char)pixel[i].b;
	}

	delete[] pixel;
	delete[] K;

	return 0;
}
