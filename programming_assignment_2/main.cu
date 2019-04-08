#include <iostream>
#include <fstream>
#include <vector>
#include <cublas_v2.h>

#include <chrono>

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))				// useful MACRO to check for errors

using namespace std;										// declare std naespace

static void HandleError( cudaError_t err, const char *file,  int line ) {			// error handling function from Dr. Mayerich
   	if (err != cudaSuccess) {
	     cout<<cudaGetErrorString( err )<<" in "<< file <<" at line "<< line;
   	}
}

void printmatrix(float *mat, int cols, int rows)						// function to print matrix
{
	for(int j = 0; j < rows; j++)								// iterate the rows
	{
		for(int i = 0; i < cols; i++)							// iterate the colums
		{
			cout << mat[i + j * cols] << " ";					// print elements
		}
		cout << endl;
	}
	cout << endl;
}

void transposematrix(float *input , float *output , int cols , int rows)			// function to transpose matrix
{
	for(int i = 0; i < rows; i++)								// iterate through rows
	{
		for(int j = 0; j < cols; j++)							// iterate through colums
		{
			output[i * cols + j] = input[i + j * rows];				// transpose 
		}
	}
}

float mse_error(float *ref, float *mat, int N_C)						// function to calculate mse error
{
	float sum = 0.0f;									// declare sum on reg
	for(int i = 0; i < N_C; i++)								// iterate through the elements
	{
		int err = ref[i] - mat[i];							// find the difference
		sum += err * err;								// calculate the square
	}
	return sum;										// return error
}


__global__
void GPU_matmul(float* A, float* B, float* C, int cols_C, int rows_C, int N_C, int cols_A)	// Naive GPU mat mul global function
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;					// find idx and idy
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= cols_C || idy >= rows_C) return;						// return the threads that are greater than rows and cols
	
	float sum = 0.0f;									// sum on reg
	for(int k = 0; k < cols_A; k++)								// itreate the row in A and col in B
	{
		sum += A[k + idy * cols_A] * B[k * cols_A + idx]; 				// compute one element of A anf one element on A and stack the values in sum
	}
	C[idx + idy * cols_C] = sum; 								// Write back to C
}

__global__ 
void GPU_Shared_matmul(float *A, float *B, float *C, int cols_C)				// Shared memory GPU Kernel
{

	extern __shared__ float sharedPtr[2 * 32 * 32];						// decalre shared memory
    	float* A_ptr = &sharedPtr[0];								// first half will have a block of A
   	float* B_ptr = &sharedPtr[32 * 32];							// second half will have a block of B
    	int width = gridDim.x*blockDim.x;							// width of the blocks to ofset 

    	float acc = 0.0f;   									// declare accumulator
    
    	int i = blockIdx.x*32 + threadIdx.x;							// calcule the thread index within block along x
    	int j = blockIdx.y*32 + threadIdx.y;							// calculate thread index within block along y

    	for (int tileIdx = 0; tileIdx < gridDim.x ; tileIdx+=1)					// iterate the blocks along the x direction
    	{
		A_ptr[threadIdx.y*32+threadIdx.x] = A[j * width + (tileIdx * 32 + threadIdx.x)];// get a block of A and move it to shared memory along x diection 
        	B_ptr[threadIdx.y*32+threadIdx.x] = B[(tileIdx * 32 + threadIdx.y) * width + i];// get a block of B and move it to shared memory along y direction
    
        	__syncthreads();                                        			// sync threads as we are writing in shared memory

        	for (int k = 0; k < 32; k++)							// itearte inside the block along x direction in A and y direction in B
        	{   
            		acc += A_ptr[threadIdx.y*32+k] * B_ptr[k*32+threadIdx.x];    		// multiply and store the reult in accumulator
        	}
    
        	__syncthreads();                                                            	// sync the threads
    	}
    	C[j * width + i ] = acc;                            					// Write in back to C
}

int main(int argc, char* argv[])
{
	if(argc != 4)										// Checking if there are 4 arguments
	{
		cout<<"Please enter 3 arguments"<<endl;						// returns this message if 3 arguments not present
		cout<<"USAGE:./executable a.mtx b.mtx c.mtx"<<endl;		
		return 1;									// end if there are not 3 arguments
	}

	string filename_1 = argv[1];								// stores the a.mtx filename to filename_1
	string filename_2 = argv[2];								// stores the b.mtx filename to filename_2

/************************** Read Matrix A ****************************/

	fstream file_1(filename_1.c_str());							// read a.mtx using file_1 handle
	unsigned int rows_A, cols_A;								// declare 2 vairable to store rows and columns of A
	file_1.read(reinterpret_cast<char *>(&rows_A),sizeof(unsigned int));			// read the rows
	cout << "Rows in A : " << rows_A << "; ";						// print it out
	file_1.read(reinterpret_cast<char *>(&cols_A),sizeof(unsigned int));			// read the colums
	cout << "Cols in A : " << cols_A << "; " << endl;					// print it out
	int N_A = rows_A * cols_A;								// store total elements of matrix in N_A

	vector<float> A_T(N_A,0);								// vector to read the column major matrix
	vector<float> A(N_A,0);									// vector to store A matrix in row major order

	file_1.read(reinterpret_cast<char *>(&A_T[0]), N_A*sizeof(float));			// read the matrix elements in A_T

//	cout << "Printing A_T matrix : " << endl;						// Uncomment this to print values of A_T
//	printmatrix(&A_T[0],cols_A, rows_A);
	
	transposematrix(&A_T[0], &A[0], cols_A, rows_A);					// Transpose A_T matrix and store it in A. Calls the function.

//	cout << "Printing A matrix : " << endl;							// Uncomment this to print values of A
//	printmatrix(&A[0],cols_A, rows_A);
	
	cout << endl ;

	file_1.close();										// Close the file handle

/************************** Read Matrix B ****************************/

	fstream file_2(filename_2.c_str());							// read b.mtx using file_2 handle
	unsigned int rows_B, cols_B;								// declare 2 vairable to store rows and columns of B
	file_2.read(reinterpret_cast<char *>(&rows_B),sizeof(unsigned int));			// read the rows
	cout << "Rows in B : " << rows_B << "; ";						// print rows
	file_2.read(reinterpret_cast<char *>(&cols_B),sizeof(unsigned int));			// read the colums
	cout << "Cols in B : " << cols_B << "; " << endl;					// print the colums

	int N_B = rows_B * cols_B;								// store the number of elements of matrix b in N_B
	vector<float> B_T(N_B,0);								// vector to store B_T
	vector<float> B(N_B,0);									// vector to store B

	file_2.read(reinterpret_cast<char *>(&B_T[0]), N_B*sizeof(float));			// read the file into B_T

//	cout << "Printing B_T matrix : " << endl;						// Uncomment to print B_T
//	printmatrix(&B_T[0],cols_B, rows_B);

	transposematrix(&B_T[0], &B[0], cols_B, rows_B);					// Transpose B_T to B

//	cout << "Printing B matrix : " << endl;							// Uncomment to print B
//	printmatrix(&B[0],cols_B, rows_B);
	
	file_2.close();
	cout << "____________________________________________________________________" << endl << endl;
	
/***************************** DECLARE C *******************************/

	vector<float> C(rows_A * cols_B, 0);							// Declare C to store A*B
	vector<float> C_T(rows_A * cols_B, 0);							// Declare C_T Sgemm gives output as C_T
	int rows_C = rows_A;									// rows of C
	int cols_C = cols_B;									// cols of C
	int N_C = rows_C * cols_C;								// Total elements in C

/******************************* CuBlas ********************************/

	cout << "Starting cublasSgemm " << endl << endl;

	float* A_gpu;										// Declare pointer on the gpu for A
	float* B_gpu;										// Declare pointer on the gpu for B
	float* C_gpu;										// Declare pointer on the GPU for C
	
	HANDLE_ERROR(cudaMalloc(&A_gpu, N_A * sizeof(float)));					// Allocate memory for A
	HANDLE_ERROR(cudaMalloc(&B_gpu, N_B * sizeof(float)));					// Allocate memory for B
	HANDLE_ERROR(cudaMalloc(&C_gpu, N_C * sizeof(float)));					// Allocate memory for C
     
	HANDLE_ERROR(cudaMemcpy(A_gpu, &A_T[0], N_A * sizeof(float),cudaMemcpyHostToDevice));	// Copy A_T to gpu
	HANDLE_ERROR(cudaMemcpy(B_gpu, &B_T[0], N_B * sizeof(float),cudaMemcpyHostToDevice));	// Copy B_T to gpu
	
	cublasHandle_t handle;									// create a cuda handle
	cublasCreate(&handle);
	
	float alf = 1;										// alpha = 1
	float bet = 0;										// beta = 0
	float *alpha = &alf;									// pointer to alpha and beta
	float *beta = &bet;

	chrono::high_resolution_clock::time_point start,stop;					// initailaize timers

	start = chrono::high_resolution_clock::now();						// record start
	cublasSgemm(handle, 				// Handle				// Launch SGEMM
	  	    CUBLAS_OP_N, CUBLAS_OP_N, 		// trans_A, trans_B
		    rows_A, cols_B, cols_A, 		// m, n, k
		    alpha,  				// *alpha
		    A_gpu, cols_A,			// *A, lda
		    B_gpu, rows_A, 			// *B, ldb
		    beta, 				// *beta
		    C_gpu, cols_A);			// *C, ldc
 
	cudaDeviceSynchronize();								// Synchronize in order to record cpu time

	stop = chrono::high_resolution_clock::now();						// record stop
	chrono::duration<double> cpu_time;
	cpu_time = chrono::duration_cast<chrono::duration<double>>(stop - start);		// calculate the time taken

 	cout << "SGEMM time taken         		: " << cpu_time.count()*1000 << " ms" << endl; 

	HANDLE_ERROR(cudaMemcpy(&C_T[0], C_gpu, rows_A * cols_B * sizeof(float),cudaMemcpyDeviceToHost));	// copy C_GPU to C_T
	
	transposematrix(&C_T[0], &C[0], cols_C, rows_C);					// transpose C_T to C

//	cout << "Printing C matrix : " << endl;							// Uncomment to print C
//	printmatrix(&C[0],cols_C, rows_C);

	cout << "**MSE Error comparison made with SGEMM Output**" << endl;			// MSE comparison will be made with output og SGEMM
	cout << "____________________________________________________________________" << endl << endl;

/******************************* CPU matmul ********************************/

	vector<float> D(N_C, 0);								// Declare another vector as I dont want to disturb C

	cout << "starting CPU Matrix mat mul  " << endl << endl;

	start = chrono::high_resolution_clock::now();						// record start

	for(int i = 0; i < rows_A; i++)								// iterate through rows
	{
		for(int j = 0; j < cols_B; j++)							// iterate through columns
		{
			float temp = 0.0f;							// use temp to store result in reg
			for(int k = 0; k < cols_A; k++)						// iterate the inner loop so that it multiplies one row of A and one col of B
			{
				temp += A[k + i * cols_A] * B[j + k * cols_B];			// Multiple one element in one row of A and one element in one col of B
			}
			D[i * cols_B + j] = temp;						// write back to D
		}
	}

	stop = chrono::high_resolution_clock::now();						// record stop

	cpu_time = chrono::duration_cast<chrono::duration<double>>(stop - start);		// calculate time taken
 
	cout << "CPU Matrix mul time taken         	: " << cpu_time.count()*1000 << " ms" << endl; 

//	cout << "Printing D matrix : " << endl;							// Use to print D
//	printmatrix(&D[0],cols_C, rows_C);
//	cout << endl;
	
	float cpu_error = mse_error(&C[0], &D[0], N_C);						// compare C and D to find MSE error. Call the function

	cout << "CPU Matrix mul mse error		: " << cpu_error << endl;		
	cout << "____________________________________________________________________" << endl << endl;

/**************************** CPU Blocked matmul *****************************/

	cout << "starting CPU Blocked Matrix mat mul  " << endl << endl;

if(rows_A >= 32 && rows_B >= 32 && cols_A >= 32 && cols_B >= 32)				// compute only if sizes are greater than 32
{

	int B_size = 32;									// size of block. I dono why I chose 32. Could have chose 51 based on cache size.
												// 2048 and 4096 are multiples of 32
	memset(&D[0], 0, cols_C * rows_C * sizeof(float));					// Clear D from previous calculation. Reusing D
	start = chrono::high_resolution_clock::now();

	for(int i = 0; i < rows_A; i += B_size)							// first three loops iterates blocks
	    for(int j = 0; j < cols_B; j += B_size)
	        for(int k = 0; k < cols_A; k += B_size)
		    for(int ii = i ; ii < i + B_size; ii++)					// next thre loops iterates inside the blocks
		        for(int jj = j; jj < j + B_size; jj++)
			    for(int kk = k; kk < k + B_size; kk++)
				D[jj + ii * cols_B] += A[ii * cols_A + kk] * B[jj + kk * cols_B];	// calculate elements inside the block

	stop = chrono::high_resolution_clock::now();						// record stop

	cpu_time = chrono::duration_cast<chrono::duration<double>>(stop - start);		// calculate time taken
 
	cout << "CPU Blocked Matrix mul time taken 	: " << cpu_time.count()*1000 << " ms" << endl; 

	float blocked_cpu_error = mse_error(&C[0], &D[0], N_C);					// compare C and D for MSE error
	cout << "CPU Blocked matrix mul mse error 	: " << blocked_cpu_error << endl;	
}
else
{
	cout << "cannot compute for sizes less than 32 " << endl;
}
	cout << "____________________________________________________________________" << endl << endl;
 
/******************************* GPU matmul ********************************/

	cout << "starting GPU Matrix mat mul  " << endl << endl;

	cudaEvent_t begin, end;									// 2 events to record time

	cudaEventCreate(&begin);								// create the 2 event
	cudaEventCreate(&end);

	float gpu_time = 0.0f;									// declare a variable to store time in milliseconds

	dim3 threads(32, 32);									// launching 2D blocks of 32 * 32		
	dim3 blocks(cols_C / 32, rows_C / 32);							// launch 2D grid

if(rows_A >= 32 && rows_B >= 32 && cols_A >= 32 && cols_B >= 32)				// compute only if sizes are greater than 32
{

	memset(&D[0], 0, cols_C * rows_C * sizeof(float));					// Clear D

	HANDLE_ERROR(cudaMemcpy(A_gpu, &A[0], rows_A * cols_A * sizeof(float),cudaMemcpyHostToDevice)); // copy A to GPU because before I had A_T in gpu
	HANDLE_ERROR(cudaMemcpy(B_gpu, &B[0], rows_B * cols_B * sizeof(float),cudaMemcpyHostToDevice));	// copy B to GPU because before I had B_T in gpu  

	cudaEventRecord(begin);									// send the start event to stream

	GPU_matmul<<<blocks, threads>>>(A_gpu, B_gpu , C_gpu, cols_C, rows_C, N_C, cols_A);	// Call the GPU Kernel

	cudaEventRecord(end);									// send the stop event to stream
	cudaEventSynchronize(end);								// wait till end occurs

	cudaEventElapsedTime(&gpu_time, begin, end);						// store the time
	cout << "GPU Matrix mul time taken		: " << gpu_time <<" ms" << endl; 	// output gpu time

	HANDLE_ERROR(cudaMemcpy(&D[0], C_gpu, rows_A * cols_B * sizeof(float),cudaMemcpyDeviceToHost));	// copy back result to D

	float gpu_error = mse_error(&C[0], &D[0], N_C);						// Calculate mse error
	cout << "GPU Matrix mul mse error 		: " << gpu_error << endl;	
}
else
{
	cout << "cannot compute for sizes less than 32 " << endl;
}
	cout << "____________________________________________________________________" << endl << endl;
	
/**************************** GPU Shared matmul *****************************/

	cout << "starting GPU Shared Matrix mat mul  " << endl << endl;

if(rows_A >= 32 && rows_B >= 32 && cols_A >= 32 && cols_B >= 32)				// compute only if sizes are greater than 32
{
	size_t shared_mem_size = 2 * 32 * 32 * sizeof(float);					// Declare the shared memory size
	
	cudaEventRecord(begin);									// send the start event to stream

        GPU_Shared_matmul<<<blocks, threads, shared_mem_size>>>(A_gpu, B_gpu, C_gpu, cols_C);	// launch the GPU shared matmul kernel

	cudaEventRecord(end);									// send the stop event to stream
	cudaEventSynchronize(end);								// wait till end occurs

	cudaEventElapsedTime(&gpu_time, begin, end);						// store the time
	cout << "GPU Shared Matrix mul time taken	: " << gpu_time <<" ms" << endl; 	// output gpu time

	memset(&D[0], 0, N_C * sizeof(float));							// Clear the previous value of D
	HANDLE_ERROR(cudaMemcpy(&D[0], C_gpu, N_C * sizeof(float), cudaMemcpyDeviceToHost));	// copy the result to D

	float gpu_shared_error = mse_error(&C[0], &D[0], N_C);					// Calculate the mse error

	cout << "gpu shared matrix mul mse error	: " << gpu_shared_error << endl;	
}
else
{
	cout << "cannot compute for sizes less than 32 " << endl;
}
	cout << "____________________________________________________________________" << endl << endl;

/********************************* GFLOPS CALC ***********************************/

	int memfetch = 2 * 4; 									// I considered 0 because I'm fetch only from shared and not global
	int FLOPS = 2;										// one add and one multiply
	float temp =  memfetch / FLOPS;
	float max_bandwidth = 732;								// 732 GFLOPS from datasheet
	
	float used_bandwidth = max_bandwidth / temp;								// used bandwidth
	cout << "Used bandwidth = " << used_bandwidth << "GFLOPS approx did not take in to account shared memory(so more than this. Out of time)" << endl;
	cout << "____________________________________________________________________" << endl << endl;

/********************************* Write C ***********************************/
	

	string filename_3 = argv[3];								// stores the c.mtx filename to filename_3
	cout << "Writing C in output filename " << filename_3 << endl << endl;
	ofstream file_3(filename_3.c_str());

	file_3.write(reinterpret_cast<char *>(&rows_C),sizeof(unsigned int));			// read the rows
	file_3.write(reinterpret_cast<char *>(&cols_C),sizeof(unsigned int));			// read the colums

	file_3.write(reinterpret_cast<char *>(&C_T[0]), N_A*sizeof(float));			// read the matrix elements in A_T
	cout << "I wrote C in column major order (did not test opening C. Out of time)" << endl;
	cout << "____________________________________________________________________" << endl << endl;

/********************************* Cleanup ***********************************/

	cudaFree(A_gpu);									// Free GPU memeory
	cudaFree(B_gpu);									// Vectors clears up automatically
	cudaFree(C_gpu);

	return 0;
}
