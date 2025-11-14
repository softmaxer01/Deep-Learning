#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <fstream>
#include <iomanip>
#include "include/main.h"

using namespace std;

// Simple JSON-like output functions
string escape_json_string(const string& str) {
    string result;
    for (char c : str) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: 
                if (c >= 32 && c <= 126) {
                    result += c;
                } else {
                    result += "\\u" + to_string(static_cast<unsigned char>(c));
                }
                break;
        }
    }
    return result;
}

string vector_to_json(const vector<int>& vec) {
    stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) ss << ",";
        ss << vec[i];
    }
    ss << "]";
    return ss.str();
}

string merges_to_json(const map<pair<int,int>, int>& merge_table) {
    stringstream ss;
    ss << "[";
    bool first = true;
    for (const auto& merge : merge_table) {
        if (!first) ss << ",";
        first = false;
        
        ss << "{";
        ss << "\"pair\":[" << merge.first.first << "," << merge.first.second << "],";
        ss << "\"result\":" << merge.second;
        ss << "}";
    }
    ss << "]";
    return ss.str();
}

string vocab_to_json(int vocab_size) {
    stringstream ss;
    ss << "[";
    for (int i = 0; i < vocab_size; ++i) {
        if (i > 0) ss << ",";
        
        ss << "{";
        ss << "\"id\":" << i << ",";
        
        if (i < 256) {
            // Base character tokens
            unsigned char ch = static_cast<unsigned char>(i);
            if (ch >= 32 && ch <= 126 && ch != '"' && ch != '\\') {
                ss << "\"char\":\"" << static_cast<char>(ch) << "\"";
            } else {
                ss << "\"char\":\"\\\\x" << hex << static_cast<int>(ch) << dec << "\"";
            }
        } else {
            // Merged tokens
            ss << "\"char\":\"tok_" << i << "\"";
        }
        ss << "}";
    }
    ss << "]";
    return ss.str();
}

vector<int> string_to_bytes(const string& text) {
    vector<int> bytes;
    for (char c : text) {
        bytes.push_back(static_cast<unsigned char>(c));
    }
    return bytes;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <command> [args...]" << endl;
        cerr << "Commands:" << endl;
        cerr << "  train <text> <vocab_size>" << endl;
        cerr << "  encode <text> <model_file>" << endl;
        cerr << "  decode <tokens> <model_file>" << endl;
        cerr << "  info <model_file>" << endl;
        return 1;
    }
    
    string command = argv[1];
    
    try {
        if (command == "train") {
            if (argc < 4) {
                cerr << "Usage: train <text> <vocab_size>" << endl;
                return 1;
            }
            
            string text = argv[2];
            int vocab_size = stoi(argv[3]);
            
            // Convert text to bytes
            vector<int> bytes = string_to_bytes(text);
            
            // Create and train tokenizer
            bpe tokenizer(bytes, vocab_size);
            tokenizer.build_merge_table();
            tokenizer.buid_vocab();
            
            // Output results as JSON
            cout << "{";
            cout << "\"success\":true,";
            cout << "\"vocab_size\":" << vocab_size << ",";
            cout << "\"text_length\":" << text.length() << ",";
            cout << "\"merges\":" << merges_to_json(tokenizer.merge_table) << ",";
            cout << "\"vocabulary\":" << vocab_to_json(vocab_size);
            cout << "}" << endl;
            
        } else if (command == "encode") {
            if (argc < 4) {
                cerr << "Usage: encode <text> <training_text> [vocab_size]" << endl;
                return 1;
            }
            
            string text = argv[2];
            string training_text = argv[3];
            int vocab_size = (argc > 4) ? stoi(argv[4]) : 280;
            
            // Train tokenizer first
            vector<int> training_bytes = string_to_bytes(training_text);
            bpe tokenizer(training_bytes, vocab_size);
            tokenizer.build_merge_table();
            tokenizer.buid_vocab();
            
            // Encode the text
            vector<int> encoded = tokenizer.encoder(text);
            
            cout << "{";
            cout << "\"success\":true,";
            cout << "\"tokens\":" << vector_to_json(encoded) << ",";
            cout << "\"token_count\":" << encoded.size() << ",";
            cout << "\"original_length\":" << text.length() << ",";
            cout << "\"compression_ratio\":" << (1.0 - (double)encoded.size() / text.length()) * 100.0;
            cout << "}" << endl;
            
        } else if (command == "decode") {
            if (argc < 4) {
                cerr << "Usage: decode <tokens> <training_text> [vocab_size]" << endl;
                return 1;
            }
            
            string tokens_str = argv[2];
            string training_text = argv[3];
            int vocab_size = (argc > 4) ? stoi(argv[4]) : 280;
            
            // Parse tokens (simple comma-separated format)
            vector<int> tokens;
            stringstream ss(tokens_str);
            string token;
            while (getline(ss, token, ',')) {
                tokens.push_back(stoi(token));
            }
            
            // Train tokenizer first
            vector<int> training_bytes = string_to_bytes(training_text);
            bpe tokenizer(training_bytes, vocab_size);
            tokenizer.build_merge_table();
            tokenizer.buid_vocab();
            
            // Decode the tokens
            string decoded = tokenizer.decoder(tokens);
            
            cout << "{";
            cout << "\"success\":true,";
            cout << "\"text\":\"" << escape_json_string(decoded) << "\",";
            cout << "\"token_count\":" << tokens.size();
            cout << "}" << endl;
            
        } else {
            cerr << "Unknown command: " << command << endl;
            return 1;
        }
        
    } catch (const exception& e) {
        cout << "{";
        cout << "\"success\":false,";
        cout << "\"error\":\"" << escape_json_string(e.what()) << "\"";
        cout << "}" << endl;
        return 1;
    }
    
    return 0;
}
