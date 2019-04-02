#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

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
	int N_A = rows_A * cols_A;

	vector<float> A_T(N_A,0);
	vector<float> A(N_A,0);

	file_1.read(reinterpret_cast<char *>(&A_T[0]), N_A*sizeof(float));

	for(int i = 0 ; i < N_A; i++)
		cout << A_T[i] << " " ;
	cout << endl;

	for(int i = 0; i < rows_A; i++)
	{
		for(int j = 0; j < cols_A; j++)
		{
			A[i * cols_A + j] = A_T[i + j * rows_A];
		}
	}

	for(auto &i: A)
		cout << i << " ";

	cout << endl <<endl;

	file_1.close();

/************************** Read Matrix B ****************************/

	fstream file_2(filename_2.c_str());
	unsigned int rows_B, cols_B;
	file_2.read(reinterpret_cast<char *>(&rows_B),sizeof(unsigned int));
	cout << "Rows in B : " << rows_B <<endl;
	file_2.read(reinterpret_cast<char *>(&cols_B),sizeof(unsigned int));
	cout << "Cols in B : " << cols_B << endl;
	int N_B = rows_B * cols_B;
	vector<float> B_T(N_B,0);
	vector<float> B(N_B,0);

	file_2.read(reinterpret_cast<char *>(&B_T[0]), N_B*sizeof(float));

	for(int i = 0 ; i < N_B; i++)
		cout << B_T[i] << " " ;
	cout << endl;

	for(int i = 0; i < rows_B; i++)
	{
		for(int j = 0; j < cols_B; j++)
		{
			B[i * cols_B + j] = B_T[i + j * rows_B];
		}
	}

	for(auto &i: B)
		cout << i << " ";

	cout << endl <<endl;

	file_2.close();
	
	return 0;
}
