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
    public:
        bpe(std::vector<int> tokens, int num_vocab);
        std::map<std::pair<int,int>, int> get_stats();
        std::vector<std::pair<std::pair<int,int>, int>> get_stats_sorted();
};
#endif // MAIN_H
