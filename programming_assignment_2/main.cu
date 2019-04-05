#include <iostream>
#include <fstream>
#include <vector>
#include <cublas_v2.h>
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))			// useful MACRO to check for errors

using namespace std;

static void HandleError( cudaError_t err, const char *file,  int line ) {
   	if (err != cudaSuccess) {
	     cout<<cudaGetErrorString( err )<<" in "<< file <<" at line "<< line;
   	}
}

void printmatrix(float *mat, int cols, int rows)
{
	for(int j = 0; j < rows; j++)
	{
		for(int i = 0; i < cols; i++)
		{
			cout << mat[i + j * cols] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void transposematrix(float *input , float *output , int cols , int rows)
{
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			output[i * cols + j] = input[i + j * rows];
		}
	}
}

float mse_error(float *ref, float *mat, int N_C)
{
	float sum = 0.0f;
	for(int i = 0; i < N_C; i++)
	{
		int err = ref[i] - mat[i];
		sum += err * err;
	}
	return sum;
}


__global__
void GPU_matmul(float* A, float* B, float* C, int cols_C, int rows_C, int N_C, int cols_A)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	if(idx >= cols_C || idy >= rows_C) return;
	
	float sum = 0.0f;	
	for(int k = 0; k < cols_A; k++)
	{
		sum += A[k + idy * cols_A] * B[k * cols_A + idx]; 	
	}
	sum = C[idx + idy * cols_C]; 
}


int main(int argc, char* argv[])
{
	if(argc !=3)									//Checking if there are 3 arguments
	{
		cout<<"Please enter 3 arguments"<<endl;					//returns this message if 3 arguments not present
		cout<<"USAGE:./executable file_a.mtx file_b.mtx "<<endl;		
		return 1;
	}

	string filename_1 = argv[1];							// stores the ppm image name to filename
	string filename_2 = argv[2];							// stores the ppm image name to filename

/************************** Read Matrix A ****************************/

	fstream file_1(filename_1.c_str());
	unsigned int rows_A, cols_A;
	file_1.read(reinterpret_cast<char *>(&rows_A),sizeof(unsigned int));
	cout << "Rows in A : " << rows_A <<endl;
	file_1.read(reinterpret_cast<char *>(&cols_A),sizeof(unsigned int));
	cout << "Cols in A : " << cols_A << endl;
	cout << endl;
	int N_A = rows_A * cols_A;

	vector<float> A_T(N_A,0);
	vector<float> A(N_A,0);

	file_1.read(reinterpret_cast<char *>(&A_T[0]), N_A*sizeof(float));

//	cout << "Printing A_T matrix : " << endl;
//	printmatrix(&A_T[0],cols_A, rows_A);
	
	transposematrix(&A_T[0], &A[0], cols_A, rows_A);

//	cout << "Printing A matrix : " << endl;
//	printmatrix(&A[0],cols_A, rows_A);
	
	cout << endl ;

	file_1.close();

/************************** Read Matrix B ****************************/

	fstream file_2(filename_2.c_str());
	unsigned int rows_B, cols_B;
	file_2.read(reinterpret_cast<char *>(&rows_B),sizeof(unsigned int));
	cout << "Rows in B : " << rows_B <<endl;
	file_2.read(reinterpret_cast<char *>(&cols_B),sizeof(unsigned int));
	cout << "Cols in B : " << cols_B << endl;
	cout << endl;

	int N_B = rows_B * cols_B;
	vector<float> B_T(N_B,0);
	vector<float> B(N_B,0);

	file_2.read(reinterpret_cast<char *>(&B_T[0]), N_B*sizeof(float));

//	cout << "Printing B_T matrix : " << endl;
//	printmatrix(&B_T[0],cols_B, rows_B);

	transposematrix(&B_T[0], &B[0], cols_B, rows_B);

//	cout << "Printing B matrix : " << endl;
//	printmatrix(&B[0],cols_B, rows_B);
	
	file_2.close();
	
/***************************** DECLARE C *******************************/

	vector<float> C(rows_A * cols_B, 0);
	vector<float> C_T(rows_A * cols_B, 0);
	int rows_C = rows_A;
	int cols_C = cols_B;
	int N_C = rows_C * cols_C;	

/******************************* CuBlas ********************************/

	cout << "Starting cublasSgemm " << endl << endl;

	float* A_gpu;
	float* B_gpu;
	float* C_gpu;
	
	HANDLE_ERROR(cudaMalloc(&A_gpu, N_A * sizeof(float)+1));
	HANDLE_ERROR(cudaMalloc(&B_gpu, N_B * sizeof(float)+1));
	HANDLE_ERROR(cudaMalloc(&C_gpu, N_C * sizeof(float)+1));
     
	HANDLE_ERROR(cudaMemcpy(A_gpu, &A_T[0], N_A * sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(B_gpu, &B_T[0], N_B * sizeof(float),cudaMemcpyHostToDevice));
	
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	float alf = 1;
	float bet = 0;
	float *alpha = &alf;
	float *beta = &bet;

	cublasSgemm(handle, 				// Handle
	  	    CUBLAS_OP_N, CUBLAS_OP_N, 		// trans_A, trans_B
		    rows_A, cols_B, cols_A, 		// m, n, k
		    alpha,  				// *alpha
		    A_gpu, cols_A,			// *A, lda
		    B_gpu, rows_A, 			// *B, ldb
		    beta, 				// *beta
		    C_gpu, cols_A);			// *C, ldc


//	cublasSgemm(handle, 				// Handle
//		    CUBLAS_OP_N, CUBLAS_OP_N, 		// trans_A, trans_B
//		    3, 2, 3,	 			// m, n, k
//		    alpha, 				// *alpha
//		    A_gpu, 3, 				// *A, lda
//		    B_gpu, 3, 				// *B, ldb
//		    beta, 				// *beta
//		    C_gpu, 3);				// *C, ldc


	HANDLE_ERROR(cudaMemcpy(&C_T[0], C_gpu, rows_A * cols_B * sizeof(float),cudaMemcpyDeviceToHost));
	
	transposematrix(&C_T[0], &C[0], cols_C, rows_C);

//	cout << "Printing C matrix : " << endl;
//	printmatrix(&C[0],cols_C, rows_C);
//	cout << endl;

//	memset(&C[0], 0, cols_C * rows_C * sizeof(float));

/******************************* CPU matmul ********************************/

	vector<float> D(N_C, 0);
	vector<float> D_T(N_C, 0);

	cout << "starting CPU Matrix mat mul  " << endl << endl;

	for(int i = 0; i < rows_A; i++)
	{
		for(int j = 0; j < cols_B; j++)
		{
			float temp = 0.0f;
			for(int k = 0; k < cols_A; k++)
			{
				temp += A[k + i * cols_A] * B[j + k * cols_B];
			}
			D[i * cols_B + j] = temp;
		}
	}

//	cout << "Printing D matrix : " << endl;
//	printmatrix(&D[0],cols_C, rows_C);
//	cout << endl;
	
	float cpu_error = mse_error(&C[0], &D[0], N_C);

	cout << "cpu matrix mul mse error compared with sgemm is : " << cpu_error << endl << endl;	

//	memset(&C[0], 0, cols_C * rows_C * sizeof(float));

/**************************** CPU Blocked matmul *****************************/

//	int B_size_r = 51;
//	int B_size_c = 51;
	
//	if(rows_C < B_size_r) B_size_r = rows_C;
//	if(cols_C < B_size_c) B_size_c = cols_C;

	int B_size = 32;	

	cout << "I'm using size of 32 so that it can be divisible by the matrix sizes " << endl << endl;

	memset(&D[0], 0, cols_C * rows_C * sizeof(float));

	for(int i = 0; i < rows_A; i += B_size)
	    for(int j = 0; j < cols_B; j += B_size)
	        for(int k = 0; k < cols_A; k += B_size)
		    for(int ii = i ; ii < i + B_size; ii++)
		        for(int jj = j; jj < j + B_size; jj++)
			    for(int kk = k; kk < k + B_size; kk++)
				D[jj + ii * cols_B] += A[ii * cols_A + kk] * B[jj + kk * cols_B];

	float blocked_cpu_error = mse_error(&C[0], &D[0], N_C);

	cout << "cpu blocked matrix mul mse error compared with sgemm is : " << blocked_cpu_error << endl << endl;	
  
//	cout << C[0] << " " << D[0] << endl;

/******************************* GPU matmul ********************************/

	dim3 threads(32, 32);
	dim3 blocks(N_C / 32 + 1, N_C / 32 + 1);

	GPU_matmul<<<blocks, threads>>>(A_gpu, B_gpu , C_gpu, cols_C, rows_C, N_C, cols_A);

	HANDLE_ERROR(cudaMemcpy(&D_T[0], C_gpu, rows_A * cols_B * sizeof(float),cudaMemcpyDeviceToHost));

	transposematrix(&D_T[0], &D[0], cols_C, rows_C);	

	float gpu_error = mse_error(&C[0], &D[0], N_C);

	cout << "gpu matrix mul mse error compared with sgemm is : " << gpu_error << endl << endl;	
	
	HANDLE_ERROR(cudaDeviceSynchronize());

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	return 0;
}
