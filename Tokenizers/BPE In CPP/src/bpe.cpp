#include <iostream>
#include <vector>
#include <map>
#include "include/main.h"
#include<algorithm>
using namespace std;

// constructor
bpe::bpe(vector<int> input_tokens, int num_vocab)
{
    tokens = input_tokens;
    this->num_vocab = num_vocab;
}

// getting the stats of biagrams
map<pair<int,int>, int> bpe::get_stats()
{
    map<pair<int,int>, int> counts;
    for(size_t i = 0; i < tokens.size() - 1; i++) {
        pair<int,int> bigram = make_pair(tokens[i], tokens[i+1]);
        counts[bigram]++;
    }
    return counts;
}

// get stats in a sorted order
vector<pair<pair<int,int>, int>> bpe::get_stats_sorted()
{
    map<pair<int,int>, int> counts = get_stats();
    vector<pair<pair<int,int>, int>> sorted_stats(counts.begin(), counts.end());
    sort(sorted_stats.begin(), sorted_stats.end(), 
         [](const pair<pair<int,int>, int>& a, const pair<pair<int,int>, int>& b) {
             return a.second > b.second; 
         });

    return sorted_stats;
}

// merge helper function to merge two consecutive to one new idx
vector<int> bpe::merge_(std::pair<int,int> pairs,int idx){
    vector<int> new_tokens;
    size_t i = 0;
    int pair_0 = pairs.first;
    int pair_1 = pairs.second;
    while(i<tokens.size()){
        if(i<tokens.size()-1 && tokens[i] == pair_0 && tokens[i+1] == pair_1){
            new_tokens.push_back(idx);
            i+=2;
        }
        else{
            new_tokens.push_back(tokens[i]);
            i++;
        }
    }
    this->tokens = new_tokens;
    return new_tokens;
}



