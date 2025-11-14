#!/bin/bash

# BPE Tokenizer Web Setup Script

echo "🚀 Setting up BPE Tokenizer Web Interface..."
echo "=========================================="

# Check if make is available
if ! command -v make &> /dev/null; then
    echo "❌ Error: 'make' is not installed. Please install build-essential:"
    echo "   sudo apt-get install build-essential"
    exit 1
fi

# Check if g++ is available
if ! command -v g++ &> /dev/null; then
    echo "❌ Error: 'g++' is not installed. Please install g++:"
    echo "   sudo apt-get install g++"
    exit 1
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: 'python3' is not installed. Please install Python 3:"
    echo "   sudo apt-get install python3 python3-pip"
    exit 1
fi

# Create obj directory if it doesn't exist
if [ ! -d "obj" ]; then
    echo "📁 Creating obj directory..."
    mkdir obj
fi

# Compile the C++ programs
echo "🔨 Compiling C++ programs..."
make clean 2>/dev/null || true
make all

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to compile C++ programs."
    echo "Please check for compilation errors above."
    exit 1
fi

# Check if simple_api was created
if [ ! -f "simple_api" ]; then
    echo "❌ Error: simple_api executable was not created."
    exit 1
fi

echo "✅ C++ programs compiled successfully!"

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null; then
    pip install -r requirements.txt
else
    echo "❌ Error: pip is not installed. Please install pip:"
    echo "   sudo apt-get install python3-pip"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install Python dependencies."
    echo "You may need to install them manually:"
    echo "   pip3 install Flask Flask-CORS"
    exit 1
fi

echo "✅ Python dependencies installed successfully!"

# Make server.py executable
chmod +x server.py

echo ""
echo "🎉 Setup completed successfully!"
echo "=========================================="
echo ""
echo "To start the web server:"
echo "  python3 server.py"
echo ""
echo "Then open your browser to:"
echo "  http://localhost:5000"
echo ""
echo "Features:"
echo "  ✨ Interactive BPE tokenization"
echo "  📊 Real-time encoding/decoding"
echo "  📈 Vocabulary and merge visualization"
echo "  🎨 TikTokenizer-inspired dark theme"
echo "  ⚡ C++ backend for fast processing"
echo ""
