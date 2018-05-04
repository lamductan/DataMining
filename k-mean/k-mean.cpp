#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <cassert>
using namespace std;

#define MAX 1000
const char DELIMITER[2] = ",";

// Declaration
class KMean
{
private:
	int m;
	int n;
	int k;

	vector<vector<double> > data;
	vector<vector<double> > centroids;
	vector<int> label;
	vector<int> cnt;
	int nSteps;

    void getData(char* datafile);
    void vectorize(vector<vector<string> > v);

    bool isConverge(vector<int> oldLabel);

public:
	KMean();
	KMean(char* datafile, int m, int n, int k);
	void train();
	void printResult();
	void printResult1();
	int assign(vector<double> p);
	//vector<int> getResult();
	~KMean();
};

double euclidean(vector<double> p1, vector<double> p2);
vector<string> tokenize(char* s);
int argmin(vector<double> v);
//string toString(char* s);
// End of declaration

int main()
{
	KMean model("input.txt", 30, 2, 3);
	model.train();
	freopen("out.csv", "w", stdout);
	model.printResult1();
	fclose(stdout);
}

// Implementation
KMean::KMean(char* datafile, int m, int n, int k) : m(m), n(n), k(k)
{
	data.resize(m);
	label.resize(m);
	for(int i = 0; i < m; ++i)
	{
		data[i].resize(n);
		label[i] = 0;
	}

	centroids.resize(k);
	cnt.resize(k);
	for(int i = 0; i < k; ++i)
	{
		centroids[i].resize(n);
		cnt[i] = 0;
	}

	getData(datafile);
	nSteps = 0;
}

void KMean::getData(char* datafile)
{
	char s[100];
	vector<string> tmp;
	FILE* f =  fopen(datafile, "r");

	for(int i = 0; i < m; ++i)
	{
		fgets(s, MAX, f);
		tmp = tokenize(s);
		for(int j = 0; j < tmp.size(); ++j)
			data[i][j] = atof(tmp[j].c_str());
	}
    fclose(f);
}

bool KMean::isConverge(vector<int> oldLabel)
{
	bool IsConverge = true;
	for(int i = 0; i < m; ++i)
		if (label[i] != oldLabel[i])
		{
			IsConverge = false;
			break;
		}
	return IsConverge;
}

int KMean::assign(vector<double> p)
{
	int Label;
	vector<double> distance(k, 0.0);
	for(int j = 0; j < k; ++j)
	{
		distance[j] = euclidean(p, centroids[j]);
	}
	Label = argmin(distance);
	return Label;
}

void KMean::train()
{
	srand(time(NULL));
	for(int i = 0; i < k; ++i)
	{
		int idx = ((rand() % m) + (rand() % m)) % m;
		centroids[i] = data[idx];
	}

	vector<int> oldLabel(m,0);
	do {
		printf("step = %d\n", nSteps);
		for(int i = 0; i < m; ++i)
			oldLabel[i] = label[i];
		for(int i = 0; i < k; ++i)
			cnt[i] = 0;

		for(int i = 0; i < m; ++i)
		{
			label[i] = assign(data[i]);
		}

		for(int i = 0; i < k; ++i)
			for(int j = 0; j < n; ++j)
				centroids[i][j] = 0.0;

		for(int i = 0; i < m; ++i)
		{
			int c = label[i];
			for(int j = 0; j < n; ++j)
				centroids[c][j] += data[i][j];
			++cnt[c];
		}
		for(int i = 0; i < k; ++i)
			for(int j = 0; j < n; ++j)
				centroids[i][j] /= (double) cnt[i];

		printResult();
		cout << "\n";
		++nSteps;
	} while (!isConverge(oldLabel));
}

void KMean::printResult()
{
	printf("Centroids:\n");
	for(int i = 0; i < k; ++i)
	{
		printf("C[%d]: (", i);
		for(int j = 0; j < n; ++j)
			printf("%0.3f ", centroids[i][j]);
		printf(")\n");
	}
	
	printf("Label:\n");
	for(int i = 0; i < m; ++i)
		printf("%d\n", label[i]);
	
	printf("Cnt:\n");
	for(int i = 0; i < k; ++i)
		printf("%d\n", cnt[i]);
}

void KMean::printResult1()
{
	for(int i = 0; i < m; ++i)
		printf("%d,%d,%d\n", (int) data[i][0], (int) data[i][1], label[i]);
}

KMean::~KMean()
{
	data.clear();
	centroids.clear();
	label.clear();
}

vector<string> tokenize(char* s)
{
	vector<string> v;
	char* token;
	token = strtok(s, DELIMITER);
	while (token)
	{
		string tmp(token);
		v.push_back(tmp);
		token = strtok(NULL, DELIMITER);
	}
	v[v.size() - 1].pop_back();
	return v;
}

double euclidean(vector<double> p1, vector<double> p2)
{
    assert(p1.size() == p2.size());
    double res = 0.0;
    for(int i = 0; i < p1.size(); ++i)
        res += pow(p1[i] -= p2[i], 2.0);
    return sqrt(res);
}

int argmin(vector<double> v)
{
	int Argmin = 0;
	for(int i = 1; i < v.size(); ++i)
		if (v[i] < v[Argmin])
			Argmin = i;
	return Argmin;
}
// End of Implementation
