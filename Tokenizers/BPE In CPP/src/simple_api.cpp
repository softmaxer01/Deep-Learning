#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "include/main.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "{\"success\":false,\"error\":\"No command provided\"}" << endl;
        return 1;
    }
    
    string command = argv[1];
    
    try {
        if (command == "train") {
            if (argc < 4) {
                cout << "{\"success\":false,\"error\":\"Usage: train <text> <vocab_size>\"}" << endl;
                return 1;
            }
            
            string text = argv[2];
            int vocab_size = stoi(argv[3]);
            
            // Convert text to bytes
            vector<int> bytes;
            for (char c : text) {
                bytes.push_back(static_cast<unsigned char>(c));
            }
            
            // Create and train tokenizer
            bpe tokenizer(bytes, vocab_size);
            tokenizer.build_merge_table();
            tokenizer.buid_vocab();
            
            // Build merges JSON
            cout << "{";
            cout << "\"success\":true,";
            cout << "\"vocab_size\":" << vocab_size << ",";
            cout << "\"text_length\":" << text.length() << ",";
            cout << "\"merge_count\":" << tokenizer.merge_table.size() << ",";
            cout << "\"vocab_count\":" << tokenizer.vocab.size() << ",";
            
            // Output merges
            cout << "\"merges\":[";
            bool first_merge = true;
            for (const auto& merge : tokenizer.merge_table) {
                if (!first_merge) cout << ",";
                first_merge = false;
                cout << "{";
                cout << "\"pair\":[" << merge.first.first << "," << merge.first.second << "],";
                cout << "\"result\":" << merge.second;
                cout << "}";
            }
            cout << "],";
            
            // Output vocabulary
            cout << "\"vocabulary\":[";
            bool first_vocab = true;
            for (const auto& token : tokenizer.vocab) {
                if (!first_vocab) cout << ",";
                first_vocab = false;
                cout << "{";
                cout << "\"id\":" << token.first << ",";
                
                // Handle special characters safely
                unsigned char ch = token.second;
                if (ch >= 32 && ch <= 126 && ch != '"' && ch != '\\') {
                    cout << "\"char\":\"" << static_cast<char>(ch) << "\"";
                } else {
                    cout << "\"char\":\"\\\\x" << hex << static_cast<int>(ch) << dec << "\"";
                }
                cout << "}";
            }
            cout << "]";
            cout << "}" << endl;
            
        } else if (command == "encode") {
            if (argc < 4) {
                cout << "{\"success\":false,\"error\":\"Usage: encode <text> <training_text> [vocab_size]\"}" << endl;
                return 1;
            }
            
            string text = argv[2];
            string training_text = argv[3];
            int vocab_size = (argc > 4) ? stoi(argv[4]) : 280;
            
            // Train tokenizer first
            vector<int> training_bytes;
            for (char c : training_text) {
                training_bytes.push_back(static_cast<unsigned char>(c));
            }
            
            bpe tokenizer(training_bytes, vocab_size);
            tokenizer.build_merge_table();
            tokenizer.buid_vocab();
            
            // Encode the text
            vector<int> encoded = tokenizer.encoder(text);
            
            cout << "{";
            cout << "\"success\":true,";
            cout << "\"tokens\":[";
            for (size_t i = 0; i < encoded.size(); ++i) {
                if (i > 0) cout << ",";
                cout << encoded[i];
            }
            cout << "],";
            cout << "\"token_count\":" << encoded.size() << ",";
            cout << "\"original_length\":" << text.length() << ",";
            double compression_ratio = text.length() > 0 ? 
                (1.0 - (double)encoded.size() / text.length()) * 100.0 : 0.0;
            cout << "\"compression_ratio\":" << compression_ratio;
            cout << "}" << endl;
            
        } else {
            cout << "{\"success\":false,\"error\":\"Unknown command\"}" << endl;
            return 1;
        }
        
    } catch (const exception& e) {
        cout << "{\"success\":false,\"error\":\"" << e.what() << "\"}" << endl;
        return 1;
    }
    
    return 0;
}
