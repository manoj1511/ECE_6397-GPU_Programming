#include <string>
#include <iostream>
#include <cstdlib>
using namespace std;

int main(int argc, char* argv[])
{
	if(argc !=3)
	{
		cout<<"Please enter 3 arguments";
		return 1;
	}
	string filename = argv[1];
	int sigma = atoi(argv[2]);
	

	return 0;
}
