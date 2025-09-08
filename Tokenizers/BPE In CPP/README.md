# BPE Tokenizer in C++

A simple implementation of Byte Pair Encoding (BPE) tokenizer written in C++.

## What is BPE?

Byte Pair Encoding is a text compression algorithm that finds the most frequent pairs of characters in text and replaces them with new tokens. It's commonly used in natural language processing for tokenization.

## Files

- `src/main.cpp` - Main program that demonstrates the tokenizer
- `src/bpe.cpp` - BPE tokenizer implementation
- `src/filereading.cpp` - File reading utilities
- `src/include/main.h` - Header file with class definitions
- `Makefile` - Build configuration

## How to Build

```bash
make
```

## How to Run

```bash
./main
```

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
