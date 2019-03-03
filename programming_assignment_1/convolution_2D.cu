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
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))			// useful MACRO to check for errors

using namespace std;

static void HandleError( cudaError_t err, const char *file,  int line ) {
   	if (err != cudaSuccess) {
	     cout<<cudaGetErrorString( err )<<" in "<< file <<" at line "<< line;
   	}
}

struct Pixel {float r, g, b; };

__global__ void convolution_gpu_x(const Pixel *in, Pixel *mid, const float *K, const int width, const int height, const int k, const int N)
{

	int indx = threadIdx.x + blockIdx.x * blockDim.x;				// calculate the thread index

	int row = indx / width;								// calculate my row
	row++;										// inc by 1 so that it doesnt have 0

	if(indx < N && indx <= ((row*width)-k))						// only enter if indx is less than N and 
	{										// width -k for each row
		float temp1=0, temp2=0, temp3=0;					// declare temps
		for(int ref = 0; ref<k; ref++)						// loop through ref for K
		{
			temp1 += K[ref] * in[indx+ref].r;				// cal K[0]*in[0] + K[1]*in[1] ... and so on
			temp2 += K[ref] * in[indx+ref].g;				// do for r, g and b
			temp3 += K[ref] * in[indx+ref].b;				// store in temp variables
		}
		mid[indx].r = temp1;							// put it in indermidiate image
		mid[indx].g = temp2;
		mid[indx].b = temp3;
	}
/*
	__syncthreads()									// sync threads doesnt provide barrier for all threads
	if(indx < N-((k-1)*width)) //&& indx <= ((row*width)-k))
	{
		float temp1=0, temp2=0, temp3=0;
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
*/

}

__global__ void convolution_gpu_y(const Pixel *mid, Pixel *out, const float *K, const int width, const int height, const int k, const int N)
{

	int indx = threadIdx.x + blockIdx.x * blockDim.x;				// calc index

	int row = indx / width;								// calc row
	row++;
/*
	if(indx < N && indx <= ((row*width)-k))
	{
		float temp1=0, temp2=0, temp3=0;
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
*/
	if(indx < N-((k-1)*width) && indx <= ((row*width)-k))				// proceed only if my thread is not below height - k or beyond width -k
	{
		float temp1=0, temp2=0, temp3=0;					// declare temps
		for(int ref = 0; ref<k; ref++)						// loop to loop through K and mid image columns
		{
			temp1 += K[ref] * mid[indx+(ref*width)].r;			// calc K[0]*mid[0] + K[1]*mid[1] + ... so on 
			temp2 += K[ref] * mid[indx+(ref*width)].g;			// mid is indexed column wise
			temp3 += K[ref] * mid[indx+(ref*width)].b;			// if width is 5 it will go like mid[0], mid[5], mid[10] 
		}
		out[indx].r = temp1;							// store back the temps to output
		out[indx].g = temp2;
		out[indx].b = temp3;
	}
}

int main(int argc, char* argv[])
{

	if(argc !=3)									//Checking if there are 3 arguments
	{
		cout<<"Please enter 3 arguments"<<endl;					//returns this message if 3 arguments not present
		cout<<"USAGE:./executable image_file_name.ppm kernal_size "<<endl;		
		return 1;
	}

	string filename = argv[1];							// stores the ppm image name to filename
	int sigma = atoi(argv[2]);							// stores the second argument to sigma

/***************************READ FILE*****************************************/

	ifstream file(filename.c_str(), ios::binary);

	string type;									// string to store file type eg. P6 
	string comment;									// string to store comments in file
	string dustbin;									// unused dustin variable to discard comments

	file >> type; 									// read file type
	file >> comment;								// read next word after type


	while(comment.compare(0,1,"#") == 0)						// see if first character is a #
	{
		getline(file,dustbin);							// then read the entire line
		file >> comment;							// read the next word
	}
	
	int width, height, range;							// variables to store the width, height and range

	width = stoi(comment);								// value after comment store it in width
	file >> height >> range;							// continue reading for height and range

	int N = height*width;								// N to store total number of pixels availanle
	cout << endl;	
	cout << "Type 		: " << type   << endl; 					// display the headers
	cout << "height      	: " << height << endl; 
	cout << "width 		: " << width  << endl; 
	cout << "range 		: " << range  << endl; 	
	cout << "N		: " << N      << endl << endl;

 	size_t buffer_size = 3 * height * width * sizeof(unsigned char) + 1;		// calulate the size of buffer needed to allocate
  	unsigned char *buffer = new unsigned char[buffer_size];				// allocate pointer for the buffer

  	file.read((char *)buffer, N*3+1);						// read the data into buffer using the pointer
  	file.close();									// close the file

	size_t pixel_size = height * width * sizeof(Pixel) + 1;				// calculate size of pixel adding 1 to be on safer side

	Pixel *pixel_in = new Pixel[pixel_size];					// allocate pointer's to store input, mid image and output image
	Pixel *pixel_mid = new Pixel[pixel_size];
	Pixel *pixel_out = new Pixel[pixel_size];

	memset(pixel_in,  0, pixel_size);						// initalize everything to 0
	memset(pixel_mid, 0, pixel_size);
	memset(pixel_out, 0, pixel_size);

	for (int i=0; i<N; i++)								// store the rgb values in pixel data type
	{
		pixel_in[i].r = buffer[i*3+1];
		pixel_in[i].g = buffer[i*3+2];
		pixel_in[i].b = buffer[i*3+3];
	}
	
	delete[] buffer;								// i dont need the buffer anymore

/*****************************CREATE KERNEL***********************************/
	
	double coeff;									// declare variable to store coefficient
	coeff = 1/sqrt(2*sigma*sigma*pi);						// calculate the coefficient
	cout << "coefficient is : " << coeff << endl << endl;				// display the coefficient value
	
	int k = 6 * sigma;								// calculate number of elements in kernel and store it in coeff
	if(k%2==0) k++;									// if k is even, increment by 1

	int k_half = k/2;								// calculate half of k
	
	float* K = new float[k];							// create a pointer to kernel K

	float sum = 0;									// declare temp variable sum
	for(int i=-k_half; i<=k_half; i++)						// loop over from -k/2 to +k/2
	{
		K[i+k_half]=coeff*exp(-(((i-mu)*(i-mu))/(2*sigma*sigma)));		// calcuate K[i] and ofset the index to store in positive values
//		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;		// uncomment this print statement to check the values of K
		sum += K[i+k_half];							// add it to temp variable sum
	}

//	cout<<"SUM	: "  <<  sum  <<  endl;						// display the sum of K use this to normalize K to 1
//	cout <<"---------------------------------------------" <<  endl;
//	cout <<"NORMALIZED K"                                  <<  endl;
//	cout <<"---------------------------------------------" <<  endl;		// uncomment these lines to see the sum

	float sum2 = 0;
	for(int i=-k_half; i<=k_half; i++)
	{
		K[i+k_half]/=sum;							// normalize K using sum
//		cout << "k["<<i+k_half<<"]	:" << K[i+k_half]<<endl;		// uncomment this print statement to check the values of normalized K
		sum2+=K[i+k_half];							// store in sum 2
	}
	
//	cout << "\nSUM after normalizing	: " << sum2 << endl;			// display sum after normalizing. uncommnet to check


/**************************CONVOLUTION ROW WISE*******************************/

	chrono::high_resolution_clock::time_point start,stop;				// initialize the timers

	start = chrono::high_resolution_clock::now();					// record the start point
	float temp1, temp2, temp3;							// declare 3 temp variables for rgb

	for(int j=0; j<height; j++)							// loop till the end height					
	{
		for(int i=0; i<=width-k; i++)						// stops at width so that ref can fill rest of image width
		{
			temp1 = 0, temp2 = 0, temp3 = 0;				// inialize the temps to 0 after each width loop
			for(int ref=0; ref<k; ref++ )
			{
				temp1 += K[ref] * pixel_in[(i+j*width)+ref].r;		// calc K[0] * input[0] + K[1] * input[1] ... and so on
				temp2 += K[ref] * pixel_in[(i+j*width)+ref].g;		// do it for r, g and b
				temp3 += K[ref] * pixel_in[(i+j*width)+ref].b;		// and store in temp variables
			}
			
			pixel_mid[i+j*width].r = temp1;		   			// copy the values to the intermediate image.
			pixel_mid[i+j*width].g = temp2;		   
			pixel_mid[i+j*width].b = temp3;		   
		}
	}
 
/*

//	UNCOMMENT THIS BLOCK TO CHECK VALUES OF MID IMAGE

	for(int i=0; i<N; i++)
	{
		if(i%20 == 0) cout << endl;
		cout <<  pixel_mid[i].r << "  "<< pixel_mid[i].g << "  "<< pixel_mid[i].b << "  ";	
	}
	cout << endl;
*/

/***********************CONVOLUTION COLUMN WISE******************************/

	for(int j=0; j<=height-k; j++)							// stops at a hight so that ref can fill rest of image height
	{
		for(int i=0; i<width; i++)						// loop through entire width which is not necessary waste of compution		
		{
			temp1 = 0; temp2 = 0 ; temp3 = 0;				// inialize the temps to 0 after each width loop
			for(int ref=0; ref<k; ref++ )
			{	
				temp1 += K[ref] * pixel_mid[(i+j*width)+(ref*width)].r; // cal K[0] * mid[0] + K[1] * mid[1] + .. so on
				temp2 += K[ref] * pixel_mid[(i+j*width)+(ref*width)].g; // mid[0], mid[1], mid[2] .. is all along column wise
				temp3 += K[ref] * pixel_mid[(i+j*width)+(ref*width)].b; // if width is 5 the indexing is like mid[0], mid[5], mid[10]
			}
			
			pixel_out[i+j*width].r = temp1;					// store it to the output image   
			pixel_out[i+j*width].g = temp2;		   
			pixel_out[i+j*width].b = temp3;		   
		}
	}

	stop = chrono::high_resolution_clock::now();					// record the stop point

	chrono::milliseconds d;								// declare d to store time in milliseconds	
	d = chrono::duration_cast<chrono::milliseconds>(stop - start);			// calculate stop - start gives the time taken

	cout << "cpu time taken		: " << d.count() << " ms" << endl;		// display the time in ms			

/*

// 	UNCOMMENT THIS BLOCK TO CHECK VALUES OF OUTPUT IMAGE

	for(int i=0; i<N; i++)
	{
		if(i%20 == 0) cout << endl;
		cout << pixel_out[i].r << "  "<<pixel_out[i].g << "  "<<pixel_out[i].b << "  ";
	}
	cout << endl;
*/

/******************************WRITE FILE*************************************/

	ofstream wfile("output_image_cpu.ppm", ios::binary);				// create or open a file in binary mode to store the output image
 	wfile << type << endl;								// write file type
 	wfile << width << " " << height << endl  << range << endl;			// write the width, height and range 

        unsigned char *out_buffer = new unsigned char[buffer_size];			// create a pointer to buffer to write easily. doesn't work if i write directly
	
	for(int i = 0; i < N; i++)							// store the output values in the buffer
	{
		out_buffer[i*3+0] = (unsigned char)pixel_out[i].r;		
		out_buffer[i*3+1] = (unsigned char)pixel_out[i].g;
		out_buffer[i*3+2] = (unsigned char)pixel_out[i].b;
	}

/*

//	UNCOMMENT THIS BLOCK TO CHECK VALUES OF OUTPUT BUFFEER

	for(int i=0; i<N*3; i++)
	{
		if(i%(3*20) == 0) cout << endl;
		cout << (int)out_buffer[i] << "   ";
	}
	cout << endl;
*/

	wfile.write(reinterpret_cast<char *>(&out_buffer[0]), N*3);			// write the values in the buffer to the the file
	wfile.close();									// close the file

	cout << "\ndone writing cpu image" << endl << endl;

	delete[] out_buffer;								// delete the buffer to free up space

/******************************GPU KERNAL*************************************/

 	cudaDeviceProp prop;
 	cudaGetDeviceProperties(&prop,0);						// store the gpu properties in prop

 	Pixel *pixel_gpu_in, *pixel_gpu_mid, *pixel_gpu_out;				// declare gpu pointers
	float *K_gpu;

 	HANDLE_ERROR(cudaMalloc(&pixel_gpu_in , pixel_size));				// allocate memory on gpu
  	HANDLE_ERROR(cudaMalloc(&pixel_gpu_mid, pixel_size));
  	HANDLE_ERROR(cudaMalloc(&pixel_gpu_out, pixel_size));
  	HANDLE_ERROR(cudaMalloc(&K_gpu, k*sizeof(float)));

	HANDLE_ERROR(cudaMemset(pixel_gpu_in , 0, pixel_size));				// set the memory to 0
	HANDLE_ERROR(cudaMemset(pixel_gpu_mid, 0, pixel_size));
	HANDLE_ERROR(cudaMemset(pixel_gpu_out, 0, pixel_size));
	HANDLE_ERROR(cudaMemset(K_gpu, 0, k*sizeof(float)));
	memset(pixel_out, 0, height*width*sizeof(Pixel));				// reuse pixel_out by setting it to 0
  
  	HANDLE_ERROR(cudaMemcpy(pixel_gpu_in, pixel_in, pixel_size, cudaMemcpyHostToDevice));	// copy the cpu_input to gpu_input 
  	HANDLE_ERROR(cudaMemcpy(K_gpu, K, k*sizeof(float), cudaMemcpyHostToDevice));	// copy cpu_K to gpu_K
  
  	int blockDim = prop.maxThreadsDim[0];						// use max threads per block
  	int gridDim = N / blockDim + 1;							// calulate max blocks based on N and blockDim
  
  	cout << "blockDim : " << blockDim << endl;					// display threads per block, num of blocks, total number od threads
  	cout << "gridDim  : " << gridDim  << endl;
	cout << "Num threads : "<< blockDim * gridDim << endl << endl;
	
	cudaEvent_t begin, end;								// 2 events to record time

	cudaEventCreate(&begin);							// create the 2 event
	cudaEventCreate(&end);
	
	cudaEventRecord(begin);								// send the start event to stream

//	I am using 2 seperate kernel because syncthread() only syncs threads within blocks so I was getting wrong results.

// 	LAUNCH THE KERNEL TO PERFORM CONVOLUTION ALONG X AXIS
  	convolution_gpu_x<<<gridDim, blockDim>>>(pixel_gpu_in, pixel_gpu_mid, K_gpu, width, height, k, N);

  	HANDLE_ERROR(cudaMemcpy(pixel_mid, pixel_gpu_mid, pixel_size, cudaMemcpyDeviceToHost));	// copy back the intermidiate image

// 	LAUNCH THE KERNEL TO PERFORM CONVOLUTION ALONG Y AXIS
  	convolution_gpu_y<<<gridDim, blockDim>>>(pixel_gpu_mid, pixel_gpu_out, K_gpu, width, height, k, N);

	cudaEventRecord(end);								// send the stop event to stream
	cudaEventSynchronize(end);							// wait till end occurs

	float milliseconds = 0;								// declare a variable to store time in milliseconds
 	cudaEventElapsedTime(&milliseconds, begin, end);				// store the time

	cout << "gpu time taken		:" << milliseconds <<" ms" << endl;		// output the time

  	HANDLE_ERROR(cudaMemcpy(pixel_out, pixel_gpu_out, pixel_size, cudaMemcpyDeviceToHost));		// copy back the final output

/*

// 	UNCOMMENT THIS BLOCK TO CHECK FINAL VALUES

	for(int i=0; i<N; i++)
	{
		if(i%(20) == 0) cout << endl;
		cout << pixel_out[i].b << "   ";		
	}
	cout << endl;
*/

/******************************WRITE FILE*************************************/

	ofstream gfile("output_image_gpu.ppm", ios::binary);				// Same as writing in cpu. just different filename
 	gfile << type << endl;
 	gfile << width << " " << height << endl  << range << endl;

        unsigned char *gpu_buffer = new unsigned char[buffer_size];
	
	for(int i = 0; i < N; i++)
	{
		gpu_buffer[i*3+0] = (unsigned char)pixel_out[i].r;
		gpu_buffer[i*3+1] = (unsigned char)pixel_out[i].g;
		gpu_buffer[i*3+2] = (unsigned char)pixel_out[i].b;
	}
/*
	for(int i=0; i<N*3; i++)
	{
		if(i%(3*20) == 0) cout << endl;
		cout << (float)gpu_buffer[i] << "   ";
	}
	cout << endl;
*/
	gfile.write(reinterpret_cast<char *>(&gpu_buffer[0]), N*3);
	gfile.close();

	cout << "\ndone writing gpu image" << endl << endl;

	delete[] pixel_in, pixel_mid, pixel_out, gpu_buffer;				// release the memory
	delete[] K, K_gpu;
 	
	return 0;
}



