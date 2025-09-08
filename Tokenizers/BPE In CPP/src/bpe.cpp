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
    this->itr = num_vocab - 256;
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

// build merge-table.
void bpe::build_merge_table(){
    for(int i = 0; i<itr;i++){
        vector<pair<pair<int,int>, int>> stats = get_stats_sorted();
        std::pair<int,int> p =  std::make_pair(stats[0].first.first,stats[0].first.second);
        int idx = 256+i;
        vector<int> new_tokens = merge_(p,idx);
        merge_table[p] = idx;
    }
}


// building vocabulary
void bpe::buid_vocab(){
    for(int i = 0; i<256;i++){
        vocab[i] = static_cast<unsigned char>(i);
    }

    for(auto &it: merge_table){ 
        int idx = it.second;
        vocab[idx] =  vocab[it.first.first] + vocab[it.first.second];
    }
}

vector<int> bpe::encoder(string text){
    vector<int> toks;
    for(auto &ch: text){
        toks.push_back(static_cast<int>(ch));
    }
    vector<int> original_tokens = tokens;
    tokens = toks;
    
    while(true){
        std::map<std::pair<int,int>, int> stats = get_stats();
        if(stats.empty()){
            break;
        }
        std::pair<int,int> best_pair;
        bool found = false;
        int max_count = 0;
        
        for(auto &stat : stats){
            if(merge_table.find(stat.first) != merge_table.end() && stat.second > max_count){
                best_pair = stat.first;
                max_count = stat.second;
                found = true;
            }
        }
        
        if(!found){
            break;
        }
        
        int idx = merge_table[best_pair];
        merge_(best_pair, idx);
    }
    vector<int> result = tokens;
    tokens = original_tokens;
    
    return result;
}

string bpe::decoder(vector<int> tokens){
    string result = "";
    for(int token : tokens){
        if(token < 256){
            result += static_cast<char>(token);
        } else {
            bool found = false;
            for(auto &merge_entry : merge_table){
                if(merge_entry.second == token){
                    
                    vector<int> sub_tokens = {merge_entry.first.first, merge_entry.first.second};
                    result += decoder(sub_tokens); 
                    found = true;
                    break;
                }
            }
            if(!found){
                result += "�"; 
            }
        }
    }
    return result;
}



