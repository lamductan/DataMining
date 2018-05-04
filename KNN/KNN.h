#include <vector>
#include <map>
using namespace std;

class KNN
{
private:
	int nSamples;
	int nAttributes;
	int kNearestNeighbors;
	int classAttribute;

	vector<vector<double> > trainData;
	vector<map<string, double> > mapValuesAttributes;
	vector<string> classTrainData;

    void getTrainData(char* trainSet);
    void vectorize(vector<vector<string> > v);
public:
	KNN();
	KNN(char* trainSet, int m, int n, int k, int classAttribute = 0);
	string classify(char* input);
	~KNN();
};

double distance(vector<double> p1, vector<double> p2);
vector<string> tokenize(char* s);
void updateMap(vector<map<string, double> >& mapValuesAttributes, vector<string> v, int nAttributes);
string toString(char* s);
