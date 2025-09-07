#include <iostream>
#include <vector>
#include <fstream>
#include "include/main.h"
using namespace std;

int main() {
    filereading f("/home/smruti/Desktop/git repos/Deep-Learning/Tokenizers/BPE In CPP/src/test.txt");
    vector<int> bytes = f.get_bytes();
    // for(int i = 0;i<100;i++){
    //     cout<<static_cast<char> (bytes[i]);
    // }
    // cout<<endl;
    cout<<f.num_chars()<<endl;
    bpe b(bytes,10);
    vector<pair<pair<int,int>, int>> sorted_stats = b.get_stats_sorted();
    std::pair<int,int> pairs = make_pair(115,116);
    vector<int> new_tokens = b.merge_(pairs,257);
    for(auto & it: new_tokens){
        cout<<it<<" ";
    }
    cout<<endl;

    return 0;
}
