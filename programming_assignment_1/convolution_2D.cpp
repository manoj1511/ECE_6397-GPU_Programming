#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <iomanip>

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

	height = stoi(comment);								// value after comment store it in height
	file >> width >> range;								// continue reading for width and range

	int N = height*width;	
 	size_t pixel_size = height * width * sizeof(Pixel);				// calulate the size needed to allocate
	
	cout<< "size to be allocated 	: " << pixel_size << endl;
	cout << type << endl << height << endl << width << endl << range << endl; 	// display the headers

	Pixel *pixel_in = new Pixel[N];							// allocate array to store pixel input values
	Pixel *pixel_mid = new Pixel[N];						// allocate array to store pixel output values
	Pixel *pixel_out = new Pixel[N];						// allocate array to store pixel output values
	
	unsigned char temp;			
	for(int i=0; i<N; i++)
	{
		file >> temp; pixel_in[i].r = temp;					// read pixel values from file one by one and convert into float
		file >> temp; pixel_in[i].g = temp; 
		file >> temp; pixel_in[i].b = temp;
	}

	file.close();

/*****************************CREATE KERNEL***********************************/
	
	coeff = 1/sqrt(2*sigma*sigma*pi);
	cout << "coefficient is :" << coeff<<endl;
	
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

	for(int i=0; i<N; i++)
	{
		if(i==400) break;

		if(i%10 == 0) cout << endl;
		cout << pixel_in[i].r << " " << pixel_in[i].g << " " << pixel_in[i].b << " ";	
	}
	cout << endl;

	cout<<"\n------------------------------------------------------------------------\n";
/**************************CONVOLUTION ROW WISE*******************************/
/*
float temp1, temp2, temp3;

	for(int j=0; j<height; j++)
	{

		for(int i=0; i<=width-k; i++)							// stops at width so that ref can fill rest of image width
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

	cout<<"\n**************************CONVOLUTION ROW WISE*******************************\n";

	for(int i=0; i<N; i++)
	{
		if(i%20 == 0) cout << endl;
		cout<<fixed <<setprecision(2)<< pixel_mid[i].b << "	";	
	}
	cout << endl;
*/
/***********************CONVOLUTION COLUMN WISE******************************/
/*
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

	cout<<"\n***********************CONVOLUTION COLUMN WISE******************************\n";

	for(int i=0; i<N; i++)
	{
		if(i%20 == 0) cout << endl;
		cout <<setprecision(2)<< pixel_out[i].b << "	";
	}
*/
/******************************WRITE FILE*************************************/

	ofstream wfile("output_image.ppm", ios::binary);
 	wfile << type << endl;
 	wfile << height << " " << width << endl  << range << endl;

	for (int i = 0; i < N; i++) 
	{
		wfile << (unsigned char)pixel_in[i].r;
		wfile << (unsigned char)pixel_in[i].g;
		wfile << (unsigned char)pixel_in[i].b;
	}
	wfile.close();


	cout << "\n done writing" << endl;
	delete[] pixel_in, pixel_mid, pixel_out;
	delete[] K;

	return 0;
}
