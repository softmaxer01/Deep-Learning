// Simple JSON implementation for the BPE web server
// For production, use nlohmann/json library

#ifndef JSON_HPP
#define JSON_HPP

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>

namespace nlohmann {

class json {
private:
    enum Type { NULL_TYPE, BOOL_TYPE, NUMBER_TYPE, STRING_TYPE, ARRAY_TYPE, OBJECT_TYPE };
    Type type = NULL_TYPE;
    
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<json> array_value;
    std::map<std::string, json> object_value;
    
public:
    // Constructors
    json() : type(NULL_TYPE) {}
    json(bool b) : type(BOOL_TYPE), bool_value(b) {}
    json(int i) : type(NUMBER_TYPE), number_value(i) {}
    json(double d) : type(NUMBER_TYPE), number_value(d) {}
    json(const std::string& s) : type(STRING_TYPE), string_value(s) {}
    json(const char* s) : type(STRING_TYPE), string_value(s) {}
    
    // Array constructor
    static json array() {
        json j;
        j.type = ARRAY_TYPE;
        return j;
    }
    
    // Object constructor  
    static json object() {
        json j;
        j.type = OBJECT_TYPE;
        return j;
    }
    
    // Array operations
    void push_back(const json& item) {
        if (type != ARRAY_TYPE) {
            type = ARRAY_TYPE;
            array_value.clear();
        }
        array_value.push_back(item);
    }
    
    // Object operations
    json& operator[](const std::string& key) {
        if (type != OBJECT_TYPE) {
            type = OBJECT_TYPE;
            object_value.clear();
        }
        return object_value[key];
    }
    
    const json& operator[](const std::string& key) const {
        static json null_json;
        auto it = object_value.find(key);
        return (it != object_value.end()) ? it->second : null_json;
    }
    
    // Value access with default
    template<typename T>
    T value(const std::string& key, const T& default_value) const {
        auto it = object_value.find(key);
        if (it != object_value.end()) {
            return static_cast<T>(it->second);
        }
        return default_value;
    }
    
    // Type conversions
    operator bool() const { return bool_value; }
    operator int() const { return static_cast<int>(number_value); }
    operator double() const { return number_value; }
    operator std::string() const { return string_value; }
    operator std::vector<int>() const {
        std::vector<int> result;
        for (const auto& item : array_value) {
            result.push_back(static_cast<int>(item));
        }
        return result;
    }
    
    // Assignment operators
    json& operator=(const std::vector<int>& vec) {
        type = ARRAY_TYPE;
        array_value.clear();
        for (int i : vec) {
            array_value.push_back(json(i));
        }
        return *this;
    }
    
    json& operator=(const std::vector<std::string>& vec) {
        type = ARRAY_TYPE;
        array_value.clear();
        for (const std::string& s : vec) {
            array_value.push_back(json(s));
        }
        return *this;
    }
    
    // Serialization
    std::string dump() const {
        std::stringstream ss;
        serialize(ss);
        return ss.str();
    }
    
    // Simple parsing (very basic implementation)
    static json parse(const std::string& str) {
        json result;
        // This is a very simplified parser - in production use proper JSON library
        if (str.find('{') != std::string::npos) {
            result.type = OBJECT_TYPE;
            // Basic parsing for common patterns
            if (str.find("\"text\"") != std::string::npos) {
                size_t start = str.find("\"text\"") + 7;
                start = str.find('"', start) + 1;
                size_t end = str.find('"', start);
                if (end != std::string::npos) {
                    result["text"] = str.substr(start, end - start);
                }
            }
            if (str.find("\"vocab_size\"") != std::string::npos) {
                size_t start = str.find("\"vocab_size\"") + 13;
                start = str.find(':', start) + 1;
                size_t end = str.find_first_of(",}", start);
                if (end != std::string::npos) {
                    std::string num_str = str.substr(start, end - start);
                    result["vocab_size"] = std::stoi(num_str);
                }
            }
        }
        return result;
    }
    
private:
    void serialize(std::stringstream& ss) const {
        switch (type) {
            case NULL_TYPE:
                ss << "null";
                break;
            case BOOL_TYPE:
                ss << (bool_value ? "true" : "false");
                break;
            case NUMBER_TYPE:
                ss << number_value;
                break;
            case STRING_TYPE:
                ss << '"' << escape_string(string_value) << '"';
                break;
            case ARRAY_TYPE:
                ss << '[';
                for (size_t i = 0; i < array_value.size(); ++i) {
                    if (i > 0) ss << ',';
                    array_value[i].serialize(ss);
                }
                ss << ']';
                break;
            case OBJECT_TYPE:
                ss << '{';
                bool first = true;
                for (const auto& pair : object_value) {
                    if (!first) ss << ',';
                    first = false;
                    ss << '"' << escape_string(pair.first) << "\":";
                    pair.second.serialize(ss);
                }
                ss << '}';
                break;
        }
    }
    
    std::string escape_string(const std::string& str) const {
        std::string result;
        for (char c : str) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c; break;
            }
        }
        return result;
    }
};

} // namespace nlohmann

#endif // JSON_HPP

