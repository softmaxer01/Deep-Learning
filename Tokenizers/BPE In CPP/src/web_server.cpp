#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <thread>
#include <chrono>
#include <fstream>
#include <algorithm>
#include "include/main.h"
#include "include/json.hpp"
#include "include/httplib.h"

using json = nlohmann::json;
using namespace std;

class BPEWebServer {
private:
    httplib::Server server;
    unique_ptr<bpe> tokenizer;
    bool is_trained = false;
    
    // Convert string to vector of bytes
    vector<int> string_to_bytes(const string& text) {
        vector<int> bytes;
        for (char c : text) {
            bytes.push_back(static_cast<unsigned char>(c));
        }
        return bytes;
    }
    
    // Convert bytes to string for display
    string bytes_to_string(const vector<int>& bytes) {
        string result;
        for (int byte : bytes) {
            if (byte < 256) {
                result += static_cast<char>(byte);
            }
        }
        return result;
    }
    
    // Get merge operations as JSON
    json get_merges_json() {
        json merges = json::array();
        if (!tokenizer) return merges;
        
        for (const auto& merge : tokenizer->merge_table) {
            json merge_obj;
            merge_obj["pair"] = {merge.first.first, merge.first.second};
            merge_obj["result"] = merge.second;
            
            // Get string representation of tokens
            string token1 = (merge.first.first < 256) ? 
                string(1, static_cast<char>(merge.first.first)) : 
                "tok_" + to_string(merge.first.first);
            string token2 = (merge.first.second < 256) ? 
                string(1, static_cast<char>(merge.first.second)) : 
                "tok_" + to_string(merge.first.second);
            
            merge_obj["pair_str"] = {token1, token2};
            merge_obj["result_str"] = "tok_" + to_string(merge.second);
            
            merges.push_back(merge_obj);
        }
        return merges;
    }
    
    // Get vocabulary as JSON
    json get_vocab_json() {
        json vocab = json::array();
        if (!tokenizer) return vocab;
        
        for (const auto& token : tokenizer->vocab) {
            json token_obj;
            token_obj["id"] = token.first;
            token_obj["token"] = string(1, token.second);
            
            // Create readable representation
            if (token.second >= 32 && token.second <= 126) {
                token_obj["display"] = string(1, token.second);
            } else {
                token_obj["display"] = "\\x" + 
                    ((token.second < 16) ? "0" : "") + 
                    to_string(token.second);
            }
            
            vocab.push_back(token_obj);
        }
        return vocab;
    }

public:
    BPEWebServer() {
        setup_routes();
    }
    
    void setup_routes() {
        // Enable CORS
        server.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type");
            return httplib::Server::HandlerResponse::Unhandled;
        });
        
        // Handle OPTIONS requests for CORS
        server.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            return;
        });
        
        // Serve static files
        server.set_mount_point("/", "./");
        
        // API Routes
        
        // Train tokenizer
        server.Post("/api/train", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                json request_data = json::parse(req.body);
                string text = request_data["text"];
                int vocab_size = request_data.value("vocab_size", 280);
                
                // Convert text to bytes
                vector<int> bytes = string_to_bytes(text);
                
                // Create and train tokenizer
                tokenizer = make_unique<bpe>(bytes, vocab_size);
                tokenizer->build_merge_table();
                tokenizer->buid_vocab();
                is_trained = true;
                
                json response;
                response["success"] = true;
                response["message"] = "Tokenizer trained successfully";
                response["vocab_size"] = vocab_size;
                response["text_length"] = text.length();
                response["merges"] = get_merges_json();
                response["vocabulary"] = get_vocab_json();
                
                res.set_content(response.dump(), "application/json");
            } catch (const exception& e) {
                json error;
                error["success"] = false;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Encode text
        server.Post("/api/encode", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                if (!is_trained || !tokenizer) {
                    json error;
                    error["success"] = false;
                    error["error"] = "Tokenizer not trained. Please train first.";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                json request_data = json::parse(req.body);
                string text = request_data["text"];
                
                // Encode text
                vector<int> encoded = tokenizer->encoder(text);
                
                json response;
                response["success"] = true;
                response["tokens"] = encoded;
                response["token_count"] = encoded.size();
                response["original_length"] = text.length();
                response["compression_ratio"] = 
                    (1.0 - (double)encoded.size() / text.length()) * 100.0;
                
                res.set_content(response.dump(), "application/json");
            } catch (const exception& e) {
                json error;
                error["success"] = false;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Decode tokens
        server.Post("/api/decode", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                if (!is_trained || !tokenizer) {
                    json error;
                    error["success"] = false;
                    error["error"] = "Tokenizer not trained. Please train first.";
                    res.status = 400;
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                json request_data = json::parse(req.body);
                vector<int> tokens = request_data["tokens"];
                
                // Decode tokens
                string decoded = tokenizer->decoder(tokens);
                
                json response;
                response["success"] = true;
                response["text"] = decoded;
                response["token_count"] = tokens.size();
                
                res.set_content(response.dump(), "application/json");
            } catch (const exception& e) {
                json error;
                error["success"] = false;
                error["error"] = e.what();
                res.status = 500;
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Get tokenizer info
        server.Get("/api/info", [this](const httplib::Request&, httplib::Response& res) {
            json response;
            response["trained"] = is_trained;
            
            if (is_trained && tokenizer) {
                response["merges"] = get_merges_json();
                response["vocabulary"] = get_vocab_json();
                response["merge_count"] = tokenizer->merge_table.size();
                response["vocab_count"] = tokenizer->vocab.size();
            }
            
            res.set_content(response.dump(), "application/json");
        });
        
        // Health check
        server.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
            json response;
            response["status"] = "healthy";
            response["timestamp"] = chrono::duration_cast<chrono::seconds>(
                chrono::system_clock::now().time_since_epoch()).count();
            res.set_content(response.dump(), "application/json");
        });
    }
    
    void start(int port = 8080) {
        cout << "Starting BPE Web Server on port " << port << "..." << endl;
        cout << "Open http://localhost:" << port << " in your browser" << endl;
        
        if (!server.listen("0.0.0.0", port)) {
            cerr << "Failed to start server on port " << port << endl;
        }
    }
    
    void stop() {
        server.stop();
    }
};

int main() {
    BPEWebServer server;
    server.start(8080);
    return 0;
}

