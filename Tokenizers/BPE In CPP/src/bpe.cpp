#include <iostream>
#include <vector>
#include <map>
#include "include/main.h"
#include<algorithm>
using namespace std;

bpe::bpe(vector<int> input_tokens, int num_vocab)
{
    tokens = input_tokens;
    this->num_vocab = num_vocab;
}

map<pair<int,int>, int> bpe::get_stats()
{
    map<pair<int,int>, int> counts;
    for(size_t i = 0; i < tokens.size() - 1; i++) {
        pair<int,int> bigram = make_pair(tokens[i], tokens[i+1]);
        counts[bigram]++;
    }
    return counts;
}

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