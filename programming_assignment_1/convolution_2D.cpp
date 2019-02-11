#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>

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

	ifstream file(filename.c_str(), ios::binary);
	string type;
	int height,width,range;

	file >> type >> height >> width >> range;


	cout << type << endl << height << endl << width << endl << range << endl; 

	file.close();
	return 0;
}
