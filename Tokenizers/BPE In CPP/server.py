#!/usr/bin/env python3
"""
Simple web server that bridges the frontend with the C++ BPE implementation.
"""

import os
import json
import subprocess
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the compiled C++ executable
CPP_EXECUTABLE = "./simple_api"

class BPEServer:
    def __init__(self):
        self.training_text = ""
        self.vocab_size = 280
        
    def call_cpp_backend(self, command, *args):
        """Call the C++ backend and return JSON response."""
        try:
            cmd = [CPP_EXECUTABLE, command] + list(args)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"C++ backend error: {e.stderr}"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parsing error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

bpe_server = BPEServer()

@app.route('/')
def index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('.', filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "backend": "C++ BPE Implementation",
        "version": "1.0.0"
    })

@app.route('/api/train', methods=['POST'])
def train_tokenizer():
    """Train the BPE tokenizer."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        vocab_size = data.get('vocab_size', 280)
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # Store training parameters for later use
        bpe_server.training_text = text
        bpe_server.vocab_size = vocab_size
        
        # Call C++ backend
        result = bpe_server.call_cpp_backend("train", text, str(vocab_size))
        
        if result.get("success"):
            # Add some additional statistics
            result["message"] = "Tokenizer trained successfully"
            result["merge_count"] = len(result.get("merges", []))
            result["vocab_count"] = len(result.get("vocabulary", []))
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/encode', methods=['POST'])
def encode_text():
    """Encode text using the trained tokenizer."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        if not bpe_server.training_text:
            return jsonify({"success": False, "error": "Tokenizer not trained. Please train first."}), 400
        
        # Call C++ backend
        result = bpe_server.call_cpp_backend(
            "encode", 
            text, 
            bpe_server.training_text, 
            str(bpe_server.vocab_size)
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/decode', methods=['POST'])
def decode_tokens():
    """Decode tokens back to text."""
    try:
        data = request.get_json()
        tokens = data.get('tokens', [])
        
        if not tokens:
            return jsonify({"success": False, "error": "No tokens provided"}), 400
        
        if not bpe_server.training_text:
            return jsonify({"success": False, "error": "Tokenizer not trained. Please train first."}), 400
        
        # Convert tokens to comma-separated string
        tokens_str = ','.join(map(str, tokens))
        
        # Call C++ backend
        result = bpe_server.call_cpp_backend(
            "decode", 
            tokens_str, 
            bpe_server.training_text, 
            str(bpe_server.vocab_size)
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/info', methods=['GET'])
def get_tokenizer_info():
    """Get information about the current tokenizer state."""
    try:
        if not bpe_server.training_text:
            return jsonify({
                "trained": False,
                "message": "No tokenizer trained yet"
            })
        
        # Get fresh training info
        result = bpe_server.call_cpp_backend(
            "train", 
            bpe_server.training_text, 
            str(bpe_server.vocab_size)
        )
        
        if result.get("success"):
            result["trained"] = True
            result["training_text_length"] = len(bpe_server.training_text)
            result["current_vocab_size"] = bpe_server.vocab_size
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sample', methods=['GET'])
def get_sample_text():
    """Get sample text for demonstration."""
    sample_texts = [
        {
            "name": "Shakespeare Sample",
            "text": """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people."""
        },
        {
            "name": "Simple Text",
            "text": "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing tokenization."
        },
        {
            "name": "Repetitive Text",
            "text": "hello world hello world hello world hello world hello world hello world hello world"
        }
    ]
    
    return jsonify({"samples": sample_texts})

if __name__ == '__main__':
    # Check if C++ executable exists
    if not os.path.exists(CPP_EXECUTABLE):
        print(f"Error: C++ executable '{CPP_EXECUTABLE}' not found.")
        print("Please compile the C++ code first:")
        print("  make all")
        exit(1)
    
    print("Starting BPE Web Server...")
    print("Backend: C++ BPE Implementation")
    print("Frontend: Web Interface")
    print("Open http://localhost:5000 in your browser")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
