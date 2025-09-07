#include<iostream>
#include<vector>
#include<fstream>
#include "include/main.h"
using namespace std;

filereading::filereading(string file_path)
{
    ifstream inputFile;
    inputFile.open(file_path);
    if (!inputFile.is_open()){
        cout<<"File can't be opened: " << file_path << endl;
        return;
    }
    char ch;
    int byte_value;
    cnt = 0;
    while(inputFile.get(ch)){
        byte_value = static_cast<unsigned char>(ch);
        bytes.push_back(byte_value);
        cnt++;
    }
    inputFile.close();
}

vector<int> filereading::get_bytes()
{
    return bytes;
}

int filereading::num_chars()
{
    return cnt;
}

filereading::~filereading()
{
}


