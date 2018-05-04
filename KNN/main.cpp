#include <iostream>
#include <cstring>
#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cassert>
#include <cmath>
#include <set>
#include "KNN.h"
using namespace std;

#define MAX 1000

const char DELIMITER[2] = ",";

KNN::KNN() {}

KNN::KNN(char* trainSet, int m, int n, int k, int classAttribute)
{
	nSamples = m;
	nAttributes = n;
	kNearestNeighbors = k;
	this->classAttribute = classAttribute;

	getTrainData(trainSet);
}

void KNN::getTrainData(char* trainSet)
{
	char s[100];
	vector<vector<string> > v;
	vector<string> tmp;
	FILE* f =  fopen(trainSet, "r");

	mapValuesAttributes.resize(nAttributes);
	for(int i = 0; i < nSamples; ++i)
	{
		fgets(s, MAX, f);
		tmp = tokenize(s);
		v.push_back(tmp);
		updateMap(mapValuesAttributes, v[i], nAttributes);
	}
    fclose(f);
    vectorize(v);
	for(int i = 0; i < nSamples; ++i)
        v[i].clear();
    v.clear();
}

void KNN::vectorize(vector<vector<string> > v)
{
    trainData.resize(nSamples);
	classTrainData.resize(nSamples);
	for(int i = 0; i < nSamples; ++i)
    {
        for(int j = 0; j < classAttribute; ++j)
            trainData[i].push_back(mapValuesAttributes[j][v[i][j]]);
        classTrainData[i] = v[i][classAttribute];
        for(int j = classAttribute + 1; j < nAttributes; ++j)
            trainData[i].push_back(mapValuesAttributes[j][v[i][j]]);
    }
}

string KNN::classify(char* input)
{
    vector<string> strInput = tokenize(input);
    vector<double> numericInput;
    for(int i = 1; i < nAttributes; ++i)
        numericInput.push_back(mapValuesAttributes[i][strInput[i]]);
    set<pair<double, int> > distanceToTrainData;
    for(int i = 0; i < nSamples; ++i)
        distanceToTrainData.insert(pair<double, int>(distance(numericInput, trainData[i]), i));

    map<string, int> cnt;
    int i = 0;
    for(auto it = distanceToTrainData.begin(); i < kNearestNeighbors; ++it, ++i)
    {
        if (cnt.find(classTrainData[it->second]) == cnt.end())
            cnt.insert(pair<string, int>(classTrainData[it->second], 1));
        else ++cnt[classTrainData[it->second]];
    }
    cout << endl;

    int maxCnt = 0;
    string res;

    for(auto it = cnt.begin(); it != cnt.end(); ++it)
        if (it->second > maxCnt)
        {
            maxCnt = it->second;
            res = it->first;
        }
    return res;
}

KNN::~KNN()
{
	trainData.clear();
	for(int i = 0; i < nAttributes; ++i)
        mapValuesAttributes[i].clear();
	mapValuesAttributes.clear();
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

void updateMap(vector<map<string, double> >& mapValuesAttributes, vector<string> v, int nAttributes)
{
	int nValues;
	for(int i = 0; i < nAttributes; ++i)
	{
		if (mapValuesAttributes[i].find(v[i]) != mapValuesAttributes[i].end())
			mapValuesAttributes[i][v[i]] += 1;
		else
		{
			mapValuesAttributes[i].insert(pair<string, double>(v[i], 1.0));
		}
	}
}

double distance(vector<double> p1, vector<double> p2)
{
    assert(p1.size() == p2.size());
    double res = 0.0;
    for(int i = 0; i < p1.size(); ++i)
        res += (p1[i] != p2[i]);
    return res;
}

string toString(char* s)
{
    stringstream ss;
    ss << s;
    string res = ss.str();
    return res;
}

int main()
{
    int classAttribute = 0;
	KNN a("train.csv", 8124, 9, 5, classAttribute);
	FILE* f = fopen("test.csv", "r");
	char s[100];
	string testLabel, predict;
	int right = 0, wrong = 0;
	while (fgets(s, 100, f)) if (strcmp(s, "\n") != 0)
    {
        testLabel = toString(s)[classAttribute];
        predict = a.classify(s);
        cout << predict << " " << testLabel << endl;
        if (predict == testLabel)
            ++right;
        else ++wrong;
    }
    cout << "Right = " << right << endl;
    cout << "Wrong = " << wrong << endl;
    fclose(f);
}
