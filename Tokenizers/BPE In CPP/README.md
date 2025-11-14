# BPE Tokenizer in C++

A comprehensive implementation of Byte Pair Encoding (BPE) tokenizer written in C++ with an interactive web interface.

## What is BPE?

Byte Pair Encoding is a text compression algorithm that finds the most frequent pairs of characters in text and replaces them with new tokens. It's commonly used in natural language processing for tokenization in models like GPT, BERT, and others.

## Features

### C++ Core Implementation
- ⚡ Fast BPE tokenization algorithm
- 📊 Configurable vocabulary size
- 🔄 Text encoding (string to token IDs)
- 🔄 Text decoding (token IDs back to string)
- 📈 Merge table and vocabulary analysis

### Interactive Web Interface
- 🎨 TikTokenizer-inspired dark theme
- 🖥️ Real-time tokenization visualization
- 📊 Interactive vocabulary and merge exploration
- 📈 Compression statistics
- 🎯 Sample text loading
- 📱 Responsive design

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
./setup.sh
python3 server.py
```

Then open http://localhost:5000 in your browser.

### Option 2: Manual Setup

1. **Build the C++ programs:**
```bash
make all
```

2. **Install Python dependencies:**
```bash
pip3 install -r requirements.txt
```

3. **Start the web server:**
```bash
python3 server.py
```

4. **Open your browser to:**
```
http://localhost:5000
```

## Files Structure

### Core C++ Implementation
- `src/main.cpp` - Original command-line demo program
- `src/bpe.cpp` - BPE tokenizer implementation
- `src/bpe_api.cpp` - Command-line API for web interface
- `src/filereading.cpp` - File reading utilities
- `src/include/main.h` - Header file with class definitions

### Web Interface
- `index.html` - Main web interface
- `styles.css` - TikTokenizer-inspired styling
- `app.js` - Frontend JavaScript application
- `server.py` - Python Flask server (bridges web ↔ C++)
- `requirements.txt` - Python dependencies

### Build System
- `Makefile` - Build configuration for both programs
- `setup.sh` - Automated setup script

## Usage

### Command Line Interface

```bash
# Run the original demo
./main

# Use the API directly
./bpe_api train "Hello world" 280
./bpe_api encode "Hello" "Hello world hello" 280
```

### Web Interface

1. **Load or enter text** in the input area
2. **Configure vocabulary size** (256-1000)
3. **Click "Train Tokenizer"** to build the BPE model
4. **View results** in the interactive panels:
   - Visual tokenization with color-coded tokens
   - Token IDs and decoded text
   - Vocabulary browser
   - Merge operations history
   - Compression statistics

## What it does

1. Reads text from `src/input.txt`
2. Builds a BPE model with the specified vocabulary size
3. Encodes text into token IDs
4. Decodes token IDs back to text
5. Shows statistics about the most frequent character pairs

## Key Features

- Text encoding (string to token IDs)
- Text decoding (token IDs back to string)
- Configurable vocabulary size
- Public access to merge table and vocabulary for analysis

## Example Output

The program shows:
- Number of characters in input file
- Most frequent character pair and its count
- Original text
- Encoded tokens
- Decoded text 

## Requirements

- C++11 compiler
- Make build system
