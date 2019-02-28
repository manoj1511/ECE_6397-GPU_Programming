#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstring>

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
//	ifstream file("output_image.ppm", ios::binary);

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

	width = stoi(comment);								// value after comment store it in height
	file >> height >> range;								// continue reading for width and range

	int N = height*width;	

	
	cout << type << endl << "height: "<<height << endl << "width: "<<width << endl << range << endl; 	// display the headers

	

 	size_t buffer_size = 3 * height * width * sizeof(unsigned char) + 1;		// calulate the size needed to allocate
        unsigned char *buffer = new unsigned char[buffer_size];

        file.read((char *)buffer, N*3+1);
        file.close();

	size_t pixel_size = height * width * sizeof(Pixel) + 1;

	Pixel *pixel_in = new Pixel[pixel_size];
	Pixel *pixel_mid = new Pixel[pixel_size];
	Pixel *pixel_out = new Pixel[pixel_size];


	for (int i=0; i<N; i++)
	{
		pixel_in[i].r = buffer[i*3+1];
		pixel_in[i].g = buffer[i*3+2];
		pixel_in[i].b = buffer[i*3+3];
	}
	
	delete[] buffer;

/*****************************CREATE KERNEL***********************************/
	
	coeff = 1/sqrt(2*sigma*sigma*pi);
	cout << "coefficient is : " << coeff<<endl;
	
	int k = 6 * sigma;
	if(k%2==0) k++;

	int k_half = k/2;
	float sum = 0;
	
	float* K = new float[k];

	for(int i=-k_half; i<=k_half; i++)
	{
		K[i+k_half]=coeff*exp(-(((i-mu)*(i-mu))/(2*sigma*sigma)));
		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;
		sum += K[i+k_half];
	
	}
	cout<<"\nSUM	: "<<sum;

	cout <<"\n---------------------------------------------\n";
	cout <<"NORMALIZED K";
	cout <<"\n---------------------------------------------\n";

	float sum2 = 0;
	for(int i=-k_half; i<=k_half; i++)
	{
		K[i+k_half]/=sum;
		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;
		sum2+=K[i+k_half];
	
	}
	
	cout<<"\nSUM2	: "<<sum2;
/*
	for(int i=0; i<N; i++)
	{
		if(i==400) break;

		if(i%10 == 0) cout << endl;
		cout << pixel_in[i].r << " " << pixel_in[i].g << " " << pixel_in[i].b << " ";	
	}
	cout << endl;

	cout<<"\n------------------------------------------------------------------------\n";
*/
/**************************CONVOLUTION ROW WISE*******************************/

float temp1, temp2, temp3;

	for(int j=0; j<height; j++)
	{

		for(int i=0; i<=width-k; i++)						// stops at width so that ref can fill rest of image width
		{
			temp1 = 0, temp2 = 0, temp3 = 0;
			for(int ref=0; ref<k; ref++ )
			{
				temp1 += K[ref] * pixel_in[(i+j*width)+ref].r;
				temp2 += K[ref] * pixel_in[(i+j*width)+ref].g;
				temp3 += K[ref] * pixel_in[(i+j*width)+ref].b;
			}
			
			pixel_mid[i+j*width].r = temp1;		   
			pixel_mid[i+j*width].g = temp2;		   
			pixel_mid[i+j*width].b = temp3;		   
		}
	}
/*
	cout<<"\n**************************CONVOLUTION ROW WISE*******************************\n";

	for(int i=0; i<N; i++)
	{
		if(i%20 == 0) cout << endl;
		cout<<fixed <<setprecision(2)<<  pixel_mid[i].r << "  "<< pixel_mid[i].g << "  "<< pixel_mid[i].b << "  ";	
	}
	cout << endl;
*/
/***********************CONVOLUTION COLUMN WISE******************************/

	memset(pixel_out, 0, height*width*sizeof(Pixel));
	for(int j=0; j<=height-k; j++)						// stops at a hight so that ref can fill rest of image height
	{

		for(int i=0; i<width; i++)		
		{
			temp1 = 0; temp2 = 0 ; temp3 = 0;
			for(int ref=0; ref<k; ref++ )
			{	
				temp1 += K[ref] * pixel_mid[(i+j*width)+(ref*width)].r;
				temp2 += K[ref] * pixel_mid[(i+j*width)+(ref*width)].g;
				temp3 += K[ref] * pixel_mid[(i+j*width)+(ref*width)].b;
			}
			
			pixel_out[i+j*width].r = temp1;		   
			pixel_out[i+j*width].g = temp2;		   
			pixel_out[i+j*width].b = temp3;		   
		}
	}

/*
	cout<<"\n***********************CONVOLUTION COLUMN WISE******************************\n";

	for(int i=0; i<N; i++)
	{
		if(i%20 == 0) cout << endl;
		cout <<setprecision(2)<< pixel_out[i].r << "  "<<pixel_out[i].g << "  "<<pixel_out[i].b << "  ";
	}
	cout << endl;
*/
/******************************WRITE FILE*************************************/



	ofstream wfile("yaxis_512.ppm", ios::binary);
 	wfile << type << endl;
 	wfile << width << " " << height << endl  << range << endl;

        unsigned char *out_buffer = new unsigned char[buffer_size];
	
	for(int i = 0; i < N; i++)
	{
		out_buffer[i*3+0] = (unsigned char)pixel_out[i].r;
		out_buffer[i*3+1] = (unsigned char)pixel_out[i].g;
		out_buffer[i*3+2] = (unsigned char)pixel_out[i].b;
	}
/*
	for(int i=0; i<N*3; i++)
	{
		if(i%(3*20) == 0) cout << endl;
		cout <<setprecision(2)<< (int)out_buffer[i] << "   ";
	}
	cout << endl;

*/

	wfile.write(reinterpret_cast<char *>(&out_buffer[0]), N*3);
	wfile.close();


	cout << "\n done writing" << endl;
	delete[] pixel_in, pixel_mid, pixel_out, out_buffer;
	delete[] K;

	return 0;
}
