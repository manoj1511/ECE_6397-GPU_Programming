#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <chrono>


#define mu 0
#define pi 3.141
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

using namespace std;

static void HandleError( cudaError_t err, const char *file,  int line ) {
   	if (err != cudaSuccess) {
	     cout<<cudaGetErrorString( err )<<" in "<< file <<" at line "<< line;
   	}
}

struct Pixel {float r, g, b; };

//__global__ struct __align__(16) Pixel_gpu{float r, g, b; };

__global__ void convolution_gpu(Pixel *in, Pixel *mid, Pixel *out, float *K, int width, int height, int k, int N)
{

	int indx = threadIdx.x + blockIdx.x * blockDim.x;

	int row = indx / width;
	row++;

	if(indx < N && indx <= ((row*width)-k))
	{
		float temp1, temp2, temp3;
		for(int ref = 0; ref<k; ref++)		
		{
			temp1 += K[ref] * in[indx+ref].r;
			temp2 += K[ref] * in[indx+ref].g;
			temp3 += K[ref] * in[indx+ref].b;
		}
		mid[indx].r = temp1;
		mid[indx].g = temp2;
		mid[indx].b = temp3;
	}
	__syncthreads();

	if(indx < N-((k-1)*width) && indx <= ((row*width)-k))
	{
		float temp1, temp2, temp3;
		for(int ref = 0; ref<k; ref++)
		{
			temp1 += K[ref] * mid[indx+(ref*width)].r;
			temp2 += K[ref] * mid[indx+(ref*width)].g;
			temp3 += K[ref] * mid[indx+(ref*width)].b;
		}
		out[indx].r = temp1;
		out[indx].g = temp2;
		out[indx].b = temp3;
	}
}


int main(int argc, char* argv[])
{

	if(argc !=3)									//Checking if there are 3 arguments
	{
		cout<<"Please enter 3 arguments"<<endl;		//returns this message if 3 arguments not present
		cout<<"USAGE:./executable image_file_name.ppm kernal_size "<<endl;		
		return 1;
	}

	string filename = argv[1];
	int sigma = atoi(argv[2]);
	double coeff;

/***************************READ FILE*****************************************/

	ifstream file(filename.c_str(), ios::binary);

	string type;									// string to store file type eg. P6 
	string comment;									// string to store comments in file
	string dustbin;									// unused dustin variable to discard comments

	file >> type; 									// read file type
	file >> comment;								// read next word after type


	while(comment.compare(0,1,"#") == 0)			// see if first character is a #
	{
		getline(file,dustbin);						// then read the entire line
		file >> comment;							// read the next word
	}
	
	int width, height, range;						// variables to store the height, width and range

	width = stoi(comment);							// value after comment store it in width
	file >> height >> range;						// continue reading for height and range

	int N = height*width;							// N to store total number of pixels availanle
	
	cout << "Type 		: " << type   << endl; 		// display the headers
	cout << "height 	: " << height << endl; 
	cout << "width 		: " << width  << endl; 
	cout << "range 		: " << range  << endl; 	

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

	cout<<"SUM	: "  <<  sum  <<  endl;
	cout <<"---------------------------------------------" <<  endl;
	cout <<"NORMALIZED K"                                  <<  endl;
	cout <<"---------------------------------------------" <<  endl;

	float sum2 = 0;
	for(int i=-k_half; i<=k_half; i++)
	{
		K[i+k_half]/=sum;
		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;
		sum2+=K[i+k_half];
	}
	
	cout<<"\nSUM2	: "<< sum2<<endl;

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
	chrono::high_resolution_clock::time_point start,stop;

	start = chrono::high_resolution_clock::now();
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

//	memset(pixel_out, 0, height*width*sizeof(Pixel));
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

	stop = chrono::high_resolution_clock::now();
	chrono::milliseconds d;
	d = chrono::duration_cast<chrono::milliseconds>(stop - start);

	cout << "time taken	: "<<d.count()<<" ms"<<endl;

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

	ofstream wfile("output_image_cpu.ppm", ios::binary);
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

	delete[] out_buffer;

/******************************GPU KERNAL*************************************/

 	cudaDeviceProp prop;
 	cudaGetDeviceProperties(&prop,0);
  	

 	Pixel *pixel_gpu_in, *pixel_gpu_mid, *pixel_gpu_out;
	float *K_gpu;

 	HANDLE_ERROR(cudaMalloc(&pixel_gpu_in , pixel_size));
  	HANDLE_ERROR(cudaMalloc(&pixel_gpu_mid, pixel_size));
  	HANDLE_ERROR(cudaMalloc(&pixel_gpu_out, pixel_size));
  	HANDLE_ERROR(cudaMalloc(&K_gpu, k*sizeof(float)));

	HANDLE_ERROR(cudaMemset(pixel_gpu_in , 0, pixel_size));
	HANDLE_ERROR(cudaMemset(pixel_gpu_mid, 0, pixel_size));
	HANDLE_ERROR(cudaMemset(pixel_gpu_out, 0, pixel_size));
	HANDLE_ERROR(cudaMemset(K_gpu, 0, k*sizeof(float)));
	memset(pixel_out, 0, height*width*sizeof(Pixel));
  
  	HANDLE_ERROR(cudaMemcpy(pixel_gpu_in, pixel_in, pixel_size, cudaMemcpyHostToDevice));
//  	HANDLE_ERROR(cudaMemcpy(pixel_in, pixel_gpu_in, pixel_size, cudaMemcpyDeviceToHost));
  	HANDLE_ERROR(cudaMemcpy(K_gpu, K, k*sizeof(float), cudaMemcpyHostToDevice));
  
  	int blockDim = prop.maxThreadsDim[0];
  	int gridDim = N / blockDim + 1;
  
  	cout << "blockDim : " << blockDim << endl;
  	cout << "gridDim  : " << gridDim  << endl;

  	convolution_gpu<<<gridDim, blockDim>>>(pixel_gpu_in, pixel_gpu_mid, pixel_gpu_out, K_gpu, width, height, k, N);
/*
  	HANDLE_ERROR(cudaMemcpy(pixel_in, pixel_gpu_in, pixel_size, cudaMemcpyDeviceToHost));

	for(int i=0; i<N; i++)
	{
		if(i%(20) == 0) cout << endl;
		cout << pixel_out[i].b << "   ";
	}
	cout << endl;
*/
	cudaDeviceSynchronize();	

  	HANDLE_ERROR(cudaMemcpy(pixel_out, pixel_gpu_out, pixel_size, cudaMemcpyDeviceToHost));
	for(int i=0; i<N; i++)
	{
		if(i%(20) == 0) cout << endl;
		cout << pixel_out[i].b << "   ";
	}
	cout << endl;


/******************************WRITE FILE*************************************/

	ofstream gfile("output_image_gpu.ppm", ios::binary);
 	gfile << type << endl;
 	gfile << width << " " << height << endl  << range << endl;

        unsigned char *gpu_buffer = new unsigned char[buffer_size];
	
	for(int i = 0; i < N; i++)
	{
		gpu_buffer[i*3+0] = (unsigned char)pixel_out[i].r;
		gpu_buffer[i*3+1] = (unsigned char)pixel_out[i].g;
		gpu_buffer[i*3+2] = (unsigned char)pixel_out[i].b;
	}

	for(int i=0; i<N*3; i++)
	{
		if(i%(3*20) == 0) cout << endl;
		cout << (float)gpu_buffer[i] << "   ";
	}
	cout << endl;

	gfile.write(reinterpret_cast<char *>(&gpu_buffer[0]), N*3);
	gfile.close();


	cout << "\n done writing" << endl;


	delete[] pixel_in, pixel_mid, pixel_out, gpu_buffer;
	delete[] K, K_gpu;
 	
	return 0;
  	
}



