#include <iostream>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

vector<vector<char> > readData(string fileName) {
    ifstream inp(fileName, ifstream::in);
    vector<vector<char> > database;
    string str;
    int idx = 0;
    while (inp >> str) {
        database.push_back(vector<char>());
        for (int i = 0; i < str.size(); ++i)
            if (str[i] != ',')
                database[idx].push_back(str[i]);
        ++idx;
    }
    //for (int i = 0; i < database.size(); ++i)
    //    assert(database[i].size() == 9);
    return database;
}

void prepare(vector<vector<char> > database, vector<int> &classes, vector<vector<vector<int> > > &cnt) {
    classes = vector<int>(2, 0);
    cnt = vector<vector<vector<int> > >(10, vector<vector<int> > (2, vector<int>(256, 0)));
    for (int i = 0; i < database.size(); ++i) {
        vector<char> &element = database[i];
        int idx = element[0] == 'e';
        ++classes[idx];
        for (int j = 1; j < element.size(); ++j)
            ++cnt[j][idx][element[j]];
    }
}

int predict(vector<int> &classes, vector<vector<vector<int> > > &cnt, vector<char> &element) {
    int best = -1;
    double rate = -1;
    int N = classes[0] + classes[1];
    for (int idx = 0; idx < 2; ++idx) {
        double conf = (double)classes[idx] / (double)N;
        for (int i = 1; i < element.size(); ++i)
            conf *= (double)cnt[i][idx][element[i]] / (double)classes[idx];
        if (conf > rate) {
            best = idx;
            rate = conf;
        }
    }
    //assert((best != -1) and (rate != -1));
    return best;
}

double evaluate(vector<vector<char> > train, vector<vector<char> > test) {
    vector<int> classes;
    vector<vector<vector<int> > > cnt;
    prepare(train, classes, cnt);
    int total = 0;
    for (int i = 0; i < test.size(); ++i) {
        int idx = test[i][0] == 'e';
        int pre = predict(classes, cnt, test[i]);
        for (int j = 1; j < test[i].size(); ++j)
            cout << test[i][j] << " ";
        cout << "predict: " << ((pre == 1) ? "e" : "p") << endl;
        if (idx == pre)
            ++total;
    }
    return (double)total / (double)test.size();
}

int main() {
    vector<vector<char> > train = readData("train.csv");
    vector<vector<char> > test = readData("test.csv");
    double accuracy = evaluate(train, test) * 100.0;
    cout << "Accuracy on test set: " << accuracy << "%";
}
