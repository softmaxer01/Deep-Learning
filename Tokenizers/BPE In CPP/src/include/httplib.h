// Simple HTTP server header - we'll use a lightweight implementation
// For production, you might want to use a more robust library like cpp-httplib

#ifndef HTTPLIB_H
#define HTTPLIB_H

#include <string>
#include <map>
#include <functional>
#include <thread>
#include <iostream>
#include <sstream>
#include <vector>

namespace httplib {

struct Request {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::string body;
};

struct Response {
    int status = 200;
    std::map<std::string, std::string> headers;
    std::string body;
    
    void set_content(const std::string& content, const std::string& content_type) {
        body = content;
        headers["Content-Type"] = content_type;
        headers["Content-Length"] = std::to_string(content.length());
    }
    
    void set_header(const std::string& key, const std::string& value) {
        headers[key] = value;
    }
};

class Server {
public:
    enum class HandlerResponse {
        Handled,
        Unhandled
    };
    
    using Handler = std::function<void(const Request&, Response&)>;
    using PreRoutingHandler = std::function<HandlerResponse(const Request&, Response&)>;
    
private:
    std::map<std::string, Handler> get_handlers;
    std::map<std::string, Handler> post_handlers;
    std::map<std::string, Handler> options_handlers;
    PreRoutingHandler pre_routing_handler;
    std::string mount_point;
    bool running = false;
    
public:
    void Get(const std::string& pattern, Handler handler) {
        get_handlers[pattern] = handler;
    }
    
    void Post(const std::string& pattern, Handler handler) {
        post_handlers[pattern] = handler;
    }
    
    void Options(const std::string& pattern, Handler handler) {
        options_handlers[pattern] = handler;
    }
    
    void set_pre_routing_handler(PreRoutingHandler handler) {
        pre_routing_handler = handler;
    }
    
    void set_mount_point(const std::string& path, const std::string& dir) {
        mount_point = dir;
    }
    
    bool listen(const std::string& host, int port) {
        // Simple implementation - in real scenario, use proper HTTP server
        std::cout << "Mock HTTP server listening on " << host << ":" << port << std::endl;
        std::cout << "Note: This is a simplified implementation." << std::endl;
        std::cout << "For production use, integrate with cpp-httplib or similar library." << std::endl;
        
        running = true;
        
        // Keep server running
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        return true;
    }
    
    void stop() {
        running = false;
    }
};

} // namespace httplib

#endif // HTTPLIB_H

