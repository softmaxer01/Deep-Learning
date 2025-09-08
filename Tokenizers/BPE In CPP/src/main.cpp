#include <iostream>
#include <vector>
#include <fstream>
#include "include/main.h"
using namespace std;

int main() {
    filereading f("/home/smruti/Desktop/git repos/Deep-Learning/Tokenizers/BPE In CPP/src/input.txt");
    vector<int> bytes = f.get_bytes();
    // for(int i = 0;i<100;i++){
    //     cout<<static_cast<char> (bytes[i]);
    // }
    // cout<<endl;
    cout<<f.num_chars()<<endl;
    
    bpe b(bytes,266);
    vector<pair<pair<int,int>, int>> sorted_stats = b.get_stats_sorted();
    for(auto &it: sorted_stats){
        cout<<"("<<it.first.first<<","<<it.first.second<<")"<<"--"<<it.second<<endl;    
        break;
    }
    b.build_merge_table();
    for (auto & it: b.merge_table){
        cout<<"("<<it.first.first<<","<<it.first.second<<")"<<"--"<<it.second<<endl;
    }

    return 0;
}
