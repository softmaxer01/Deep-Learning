#ifndef MAIN_H
#define MAIN_H
#include <string>
#include <vector>
#include <map>

class filereading {
    private:
        std::vector<int> bytes;
        int cnt = 0;
    public:
        filereading(std::string file_path);
        std::vector<int> get_bytes();
        int num_chars();
        ~filereading();
};

class bpe {
    private:
        std::vector<int> tokens;
        int num_vocab;
        int itr;

    public:
        // Public member variables for external access
        std::map<std::pair<int, int>, int> merge_table;
        std::map<int,unsigned char> vocab;
        
        // Constructor and methods
        bpe(std::vector<int> tokens, int num_vocab);
        std::map<std::pair<int,int>, int> get_stats();
        std::vector<std::pair<std::pair<int,int>, int>> get_stats_sorted();
        std::vector<int> merge_(std::pair<int,int> pairs,int idx);
        void build_merge_table();
        void buid_vocab();
        std::vector<int> encoder(std::string s);
        std::string decoder(std::vector<int> tokens);
};
#endif // MAIN_H
