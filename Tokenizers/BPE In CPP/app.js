// BPE Tokenizer Web Interface
// Communicates with C++ backend via Python Flask server

class BPETokenizerApp {
    constructor() {
        this.baseURL = window.location.origin;
        this.isTraining = false;
        this.isTrained = false;
        this.currentTokens = [];
        this.currentMerges = [];
        this.currentVocab = [];
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadSampleTexts();
        this.checkServerHealth();
    }
    
    initializeElements() {
        // Input elements
        this.inputText = document.getElementById('inputText');
        this.vocabSize = document.getElementById('vocabSize');
        this.showMerges = document.getElementById('showMerges');
        
        // Buttons
        this.trainButton = document.getElementById('trainTokenizer');
        this.loadSampleButton = document.getElementById('loadSample');
        this.clearButton = document.getElementById('clearText');
        
        // Display elements
        this.charCount = document.getElementById('charCount');
        this.byteCount = document.getElementById('byteCount');
        this.tokenCount = document.getElementById('tokenCount');
        this.compressionRatio = document.getElementById('compressionRatio');
        this.vocabCount = document.getElementById('vocabCount');
        
        // Result containers
        this.visualTokens = document.getElementById('visualTokens');
        this.tokenIds = document.getElementById('tokenIds');
        this.decodedText = document.getElementById('decodedText');
        this.vocabularyList = document.getElementById('vocabularyList');
        this.mergesList = document.getElementById('mergesList');
        this.statisticsView = document.getElementById('statisticsView');
        
        // Progress elements
        this.progressSection = document.getElementById('progressSection');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.progressPercent = document.getElementById('progressPercent');
    }
    
    setupEventListeners() {
        // Input text changes
        this.inputText.addEventListener('input', () => {
            this.updateInputStats();
            if (this.isTrained) {
                this.debounce(this.encodeText.bind(this), 500)();
            }
        });
        
        // Vocab size changes
        this.vocabSize.addEventListener('change', () => {
            if (this.isTrained) {
                this.showMessage('Vocabulary size changed. Please retrain the tokenizer.', 'warning');
                this.isTrained = false;
                this.updateTrainButton();
            }
        });
        
        // Button clicks
        this.trainButton.addEventListener('click', () => this.trainTokenizer());
        this.loadSampleButton.addEventListener('click', () => this.loadSampleText());
        this.clearButton.addEventListener('click', () => this.clearText());
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target));
        });
        
        // Initial stats update
        this.updateInputStats();
    }
    
    async checkServerHealth() {
        try {
            const response = await fetch(`${this.baseURL}/api/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.showMessage('Connected to C++ BPE backend', 'success');
            }
        } catch (error) {
            this.showMessage('Failed to connect to backend server', 'error');
            console.error('Health check failed:', error);
        }
    }
    
    async loadSampleTexts() {
        try {
            const response = await fetch(`${this.baseURL}/api/sample`);
            const data = await response.json();
            this.sampleTexts = data.samples || [];
        } catch (error) {
            console.error('Failed to load sample texts:', error);
            // Fallback sample texts
            this.sampleTexts = [
                {
                    name: "Simple Example",
                    text: "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing tokenization."
                }
            ];
        }
    }
    
    loadSampleText() {
        if (this.sampleTexts && this.sampleTexts.length > 0) {
            const sample = this.sampleTexts[0]; // Use first sample
            this.inputText.value = sample.text;
            this.updateInputStats();
            this.showMessage(`Loaded sample: ${sample.name}`, 'info');
        }
    }
    
    clearText() {
        this.inputText.value = '';
        this.updateInputStats();
        this.clearResults();
    }
    
    updateInputStats() {
        const text = this.inputText.value;
        const charCount = text.length;
        const byteCount = new TextEncoder().encode(text).length;
        
        this.charCount.textContent = `${charCount} characters`;
        this.byteCount.textContent = `${byteCount} bytes`;
    }
    
    async trainTokenizer() {
        const text = this.inputText.value.trim();
        if (!text) {
            this.showMessage('Please enter some text to train the tokenizer', 'error');
            return;
        }
        
        const vocabSize = parseInt(this.vocabSize.value);
        if (vocabSize < 256 || vocabSize > 1000) {
            this.showMessage('Vocabulary size must be between 256 and 1000', 'error');
            return;
        }
        
        this.isTraining = true;
        this.updateTrainButton();
        this.showProgress('Training tokenizer...', 0);
        
        try {
            // Simulate progress updates
            const progressInterval = setInterval(() => {
                const currentProgress = parseFloat(this.progressFill.style.width) || 0;
                if (currentProgress < 90) {
                    this.updateProgress(currentProgress + 10);
                }
            }, 200);
            
            const response = await fetch(`${this.baseURL}/api/train`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    vocab_size: vocabSize
                })
            });
            
            clearInterval(progressInterval);
            
            const data = await response.json();
            
            if (data.success) {
                this.isTrained = true;
                this.currentMerges = data.merges || [];
                this.currentVocab = data.vocabulary || [];
                
                this.updateProgress(100);
                this.showMessage('Tokenizer trained successfully!', 'success');
                
                // Update displays
                this.updateVocabularyDisplay();
                this.updateMergesDisplay();
                this.updateStatistics(data);
                
                // Encode the current text
                await this.encodeText();
                
                setTimeout(() => this.hideProgress(), 1000);
            } else {
                throw new Error(data.error || 'Training failed');
            }
        } catch (error) {
            this.showMessage(`Training failed: ${error.message}`, 'error');
            console.error('Training error:', error);
            this.hideProgress();
        } finally {
            this.isTraining = false;
            this.updateTrainButton();
        }
    }
    
    async encodeText() {
        const text = this.inputText.value.trim();
        if (!text || !this.isTrained) return;
        
        try {
            const response = await fetch(`${this.baseURL}/api/encode`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentTokens = data.tokens || [];
                this.updateTokenizationDisplay(data);
                await this.decodeTokens();
            } else {
                throw new Error(data.error || 'Encoding failed');
            }
        } catch (error) {
            this.showMessage(`Encoding failed: ${error.message}`, 'error');
            console.error('Encoding error:', error);
        }
    }
    
    async decodeTokens() {
        if (!this.currentTokens.length || !this.isTrained) return;
        
        try {
            const response = await fetch(`${this.baseURL}/api/decode`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ tokens: this.currentTokens })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.decodedText.textContent = data.text;
                // Now that we have the decoded text, create the token mapping and visualization
                this.createTokenTextMapping();
                this.updateHighlightedTextDisplay();
            } else {
                throw new Error(data.error || 'Decoding failed');
            }
        } catch (error) {
            this.showMessage(`Decoding failed: ${error.message}`, 'error');
            console.error('Decoding error:', error);
            
            // Even if decoding fails, create a basic visualization using the original text
            this.createFallbackTokenMapping();
            this.updateHighlightedTextDisplay();
        }
    }
    
    updateTokenizationDisplay(data) {
        // Update stats
        this.tokenCount.textContent = `${data.token_count || 0} tokens`;
        const compressionRatio = data.compression_ratio || 0;
        this.compressionRatio.textContent = `${compressionRatio.toFixed(1)}% compression`;
        
        // Store original text - mapping will be created after decoding
        this.originalText = this.inputText.value;
        
        // Update visual tokens
        this.visualTokens.innerHTML = '';
        this.currentTokens.forEach((tokenId, index) => {
            const tokenElement = document.createElement('span');
            tokenElement.className = 'token';
            tokenElement.textContent = tokenId;
            tokenElement.title = `Token ID: ${tokenId}`;
            tokenElement.dataset.tokenIndex = index;
            
            // Add special styling for certain token types
            if (tokenId < 256) {
                const char = String.fromCharCode(tokenId);
                if (char === '\n') {
                    tokenElement.className += ' newline';
                    tokenElement.textContent = '\\n';
                } else if (char.match(/\s/)) {
                    tokenElement.className += ' special';
                    tokenElement.textContent = '·';
                }
            }
            
            // Add hover event listeners for synchronization
            tokenElement.addEventListener('mouseenter', () => this.highlightToken(index));
            tokenElement.addEventListener('mouseleave', () => this.unhighlightToken(index));
            
            this.visualTokens.appendChild(tokenElement);
        });
        
        // Update token IDs display with individual hoverable elements
        this.tokenIds.innerHTML = '';
        this.currentTokens.forEach((tokenId, index) => {
            const tokenIdElement = document.createElement('span');
            tokenIdElement.className = 'token-id';
            tokenIdElement.textContent = tokenId;
            tokenIdElement.title = `Token ID: ${tokenId}`;
            tokenIdElement.dataset.tokenIndex = index;
            
            // Add hover event listeners for synchronization
            tokenIdElement.addEventListener('mouseenter', () => this.highlightToken(index));
            tokenIdElement.addEventListener('mouseleave', () => this.unhighlightToken(index));
            
            this.tokenIds.appendChild(tokenIdElement);
            
            // Add comma separator (except for last element)
            if (index < this.currentTokens.length - 1) {
                const separator = document.createElement('span');
                separator.className = 'token-separator';
                separator.textContent = ', ';
                this.tokenIds.appendChild(separator);
            }
        });
    }
    
    updateVocabularyDisplay() {
        this.vocabularyList.innerHTML = '';
        this.vocabCount.textContent = `${this.currentVocab.length} tokens`;
        
        // Show only a subset for performance (first 100 tokens)
        const displayVocab = this.currentVocab.slice(0, 100);
        
        displayVocab.forEach(token => {
            const vocabItem = document.createElement('div');
            vocabItem.className = 'vocab-item';
            
            const tokenSpan = document.createElement('span');
            tokenSpan.className = 'vocab-token';
            
            // Better display for different token types
            let displayText = token.char || `tok_${token.id}`;
            if (token.id < 256) {
                // Base character tokens
                if (token.char && token.char.length === 1) {
                    const charCode = token.char.charCodeAt(0);
                    if (charCode >= 32 && charCode <= 126) {
                        displayText = token.char;
                    } else if (charCode === 32) {
                        displayText = '␣'; // Space symbol
                    } else if (charCode === 10) {
                        displayText = '↵'; // Newline symbol
                    } else if (charCode === 9) {
                        displayText = '⇥'; // Tab symbol
                    } else {
                        displayText = `\\x${charCode.toString(16).padStart(2, '0')}`;
                    }
                } else if (token.char && token.char.startsWith('\\x')) {
                    displayText = token.char;
                }
            } else {
                // Merged tokens
                displayText = `[${token.id}]`;
            }
            
            tokenSpan.textContent = displayText;
            tokenSpan.title = `Token ID: ${token.id}, Char: ${token.char}`;
            
            const idSpan = document.createElement('span');
            idSpan.className = 'vocab-id';
            idSpan.textContent = token.id;
            
            vocabItem.appendChild(tokenSpan);
            vocabItem.appendChild(idSpan);
            this.vocabularyList.appendChild(vocabItem);
        });
        
        if (this.currentVocab.length > 100) {
            const moreItem = document.createElement('div');
            moreItem.className = 'vocab-item';
            moreItem.style.fontStyle = 'italic';
            moreItem.style.color = 'var(--text-muted)';
            moreItem.textContent = `... and ${this.currentVocab.length - 100} more tokens`;
            this.vocabularyList.appendChild(moreItem);
        }
    }
    
    updateMergesDisplay() {
        this.mergesList.innerHTML = '';
        
        this.currentMerges.forEach((merge, index) => {
            const mergeItem = document.createElement('div');
            mergeItem.className = 'merge-item';
            
            const pairDiv = document.createElement('div');
            pairDiv.className = 'merge-pair';
            
            // Helper function to display token
            const getTokenDisplay = (tokenId) => {
                if (tokenId < 256) {
                    const char = String.fromCharCode(tokenId);
                    if (tokenId >= 32 && tokenId <= 126) {
                        return char;
                    } else if (tokenId === 32) {
                        return '␣';
                    } else if (tokenId === 10) {
                        return '↵';
                    } else if (tokenId === 9) {
                        return '⇥';
                    } else {
                        return `\\x${tokenId.toString(16).padStart(2, '0')}`;
                    }
                } else {
                    return `[${tokenId}]`;
                }
            };
            
            const stepSpan = document.createElement('span');
            stepSpan.className = 'merge-step';
            stepSpan.textContent = `${index + 1}. `;
            stepSpan.style.color = 'var(--text-muted)';
            stepSpan.style.fontSize = '0.75rem';
            
            const token1 = document.createElement('span');
            token1.className = 'vocab-token';
            token1.textContent = getTokenDisplay(merge.pair[0]);
            token1.title = `Token ID: ${merge.pair[0]}`;
            
            const arrow = document.createElement('span');
            arrow.className = 'merge-arrow';
            arrow.textContent = ' + ';
            
            const token2 = document.createElement('span');
            token2.className = 'vocab-token';
            token2.textContent = getTokenDisplay(merge.pair[1]);
            token2.title = `Token ID: ${merge.pair[1]}`;
            
            const resultArrow = document.createElement('span');
            resultArrow.className = 'merge-arrow';
            resultArrow.textContent = ' → ';
            
            const result = document.createElement('span');
            result.className = 'merge-result';
            result.textContent = `[${merge.result}]`;
            result.title = `New token ID: ${merge.result}`;
            
            pairDiv.appendChild(stepSpan);
            pairDiv.appendChild(token1);
            pairDiv.appendChild(arrow);
            pairDiv.appendChild(token2);
            pairDiv.appendChild(resultArrow);
            pairDiv.appendChild(result);
            
            mergeItem.appendChild(pairDiv);
            this.mergesList.appendChild(mergeItem);
        });
        
        if (this.currentMerges.length === 0) {
            const emptyMessage = document.createElement('div');
            emptyMessage.className = 'merge-item';
            emptyMessage.style.fontStyle = 'italic';
            emptyMessage.style.color = 'var(--text-muted)';
            emptyMessage.textContent = 'No merges yet. Train the tokenizer first.';
            this.mergesList.appendChild(emptyMessage);
        }
    }
    
    updateStatistics(data) {
        this.statisticsView.innerHTML = '';
        
        const stats = [
            { label: 'Vocabulary Size', value: data.vocab_size || 0 },
            { label: 'Text Length', value: data.text_length || 0 },
            { label: 'Merge Operations', value: data.merge_count || 0 },
            { label: 'Total Tokens', value: data.vocab_count || 0 }
        ];
        
        stats.forEach(stat => {
            const statCard = document.createElement('div');
            statCard.className = 'stat-card';
            
            const valueSpan = document.createElement('span');
            valueSpan.className = 'stat-value';
            valueSpan.textContent = stat.value;
            
            const labelSpan = document.createElement('span');
            labelSpan.className = 'stat-label';
            labelSpan.textContent = stat.label;
            
            statCard.appendChild(valueSpan);
            statCard.appendChild(labelSpan);
            this.statisticsView.appendChild(statCard);
        });
    }
    
    switchTab(tabBtn) {
        const tabName = tabBtn.dataset.tab;
        const tabContainer = tabBtn.closest('.result-section');
        
        // Update tab buttons
        tabContainer.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        tabBtn.classList.add('active');
        
        // Update tab panes
        tabContainer.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        
        const targetPane = tabContainer.querySelector(`#${tabName}Tab`);
        if (targetPane) {
            targetPane.classList.add('active');
        }
    }
    
    updateTrainButton() {
        if (this.isTraining) {
            this.trainButton.textContent = 'Training...';
            this.trainButton.disabled = true;
        } else {
            this.trainButton.textContent = this.isTrained ? 'Retrain Tokenizer' : 'Train Tokenizer';
            this.trainButton.disabled = false;
        }
    }
    
    showProgress(text, percent) {
        this.progressSection.style.display = 'block';
        this.progressText.textContent = text;
        this.updateProgress(percent);
    }
    
    updateProgress(percent) {
        this.progressFill.style.width = `${percent}%`;
        this.progressPercent.textContent = `${Math.round(percent)}%`;
    }
    
    hideProgress() {
        this.progressSection.style.display = 'none';
    }
    
    clearResults() {
        this.visualTokens.innerHTML = '';
        this.tokenIds.innerHTML = '';
        this.decodedText.textContent = '';
        this.tokenCount.textContent = '0 tokens';
        this.compressionRatio.textContent = '0% compression';
        
        // Clear highlighted text display
        const highlightedTextContainer = document.getElementById('highlightedText');
        if (highlightedTextContainer) {
            highlightedTextContainer.remove();
        }
        
        // Clear token text mapping
        this.tokenTextMapping = [];
        this.originalText = '';
        this.reconstructedText = '';
    }
    
    createTokenTextMapping() {
        // Use the already decoded text from the decode API call
        // We'll use a simpler approach: divide the decoded text among tokens proportionally
        
        console.log('Creating token text mapping...');
        console.log('decodedText element:', this.decodedText);
        console.log('decodedText content:', this.decodedText?.textContent);
        
        if (!this.decodedText || !this.decodedText.textContent) {
            console.log('No decoded text available');
            return;
        }
        
        const fullDecodedText = this.decodedText.textContent;
        this.reconstructedText = fullDecodedText;
        this.tokenTextMapping = [];
        
        // Simple approach: divide the text evenly among tokens
        // This isn't perfect but gives a reasonable approximation
        const textLength = fullDecodedText.length;
        const tokenCount = this.currentTokens.length;
        
        let currentPos = 0;
        
        for (let i = 0; i < this.currentTokens.length; i++) {
            const tokenId = this.currentTokens[i];
            
            // Calculate approximate segment for this token
            let segmentLength;
            if (i === this.currentTokens.length - 1) {
                // Last token gets remaining text
                segmentLength = textLength - currentPos;
            } else {
                // Estimate based on remaining tokens and text
                const remainingTokens = tokenCount - i;
                const remainingText = textLength - currentPos;
                segmentLength = Math.max(1, Math.floor(remainingText / remainingTokens));
            }
            
            const startPos = currentPos;
            const endPos = Math.min(currentPos + segmentLength, textLength);
            const tokenText = fullDecodedText.substring(startPos, endPos);
            
            this.tokenTextMapping.push({
                tokenIndex: i,
                tokenId: tokenId,
                text: tokenText,
                startPos: startPos,
                endPos: endPos
            });
            
            currentPos = endPos;
        }
        
        console.log('Original text:', this.originalText);
        console.log('Decoded text:', fullDecodedText);
        console.log('Token mapping:', this.tokenTextMapping);
    }
    
    createFallbackTokenMapping() {
        // Fallback method when decoding fails - use original text
        console.log('Creating fallback token mapping using original text...');
        
        if (!this.originalText || !this.currentTokens.length) {
            console.log('No original text or tokens available for fallback');
            return;
        }
        
        this.reconstructedText = this.originalText;
        this.tokenTextMapping = [];
        
        // Simple approach: divide the original text evenly among tokens
        const textLength = this.originalText.length;
        const tokenCount = this.currentTokens.length;
        
        let currentPos = 0;
        
        for (let i = 0; i < this.currentTokens.length; i++) {
            const tokenId = this.currentTokens[i];
            
            // Calculate approximate segment for this token
            let segmentLength;
            if (i === this.currentTokens.length - 1) {
                // Last token gets remaining text
                segmentLength = textLength - currentPos;
            } else {
                // Estimate based on remaining tokens and text
                const remainingTokens = tokenCount - i;
                const remainingText = textLength - currentPos;
                segmentLength = Math.max(1, Math.floor(remainingText / remainingTokens));
            }
            
            const startPos = currentPos;
            const endPos = Math.min(currentPos + segmentLength, textLength);
            const tokenText = this.originalText.substring(startPos, endPos);
            
            this.tokenTextMapping.push({
                tokenIndex: i,
                tokenId: tokenId,
                text: tokenText,
                startPos: startPos,
                endPos: endPos
            });
            
            currentPos = endPos;
        }
        
        console.log('Fallback token mapping created:', this.tokenTextMapping);
    }
    
    
    updateHighlightedTextDisplay() {
        console.log('Updating highlighted text display...');
        console.log('reconstructedText:', this.reconstructedText);
        
        if (!this.reconstructedText) {
            console.log('No reconstructed text available');
            return;
        }
        
        // Create a new container for highlighted text if it doesn't exist
        let highlightedTextContainer = document.getElementById('highlightedText');
        console.log('Highlighted text container found:', highlightedTextContainer);
        
        if (!highlightedTextContainer) {
            // Add it to the visual tab
            const visualTab = document.getElementById('visualTab');
            console.log('Visual tab found:', visualTab);
            
            if (visualTab) {
                const separator = document.createElement('div');
                separator.style.margin = '16px 0';
                separator.style.borderTop = '1px solid var(--border-primary)';
                
                const label = document.createElement('div');
                label.textContent = 'Tokenized Text (hover over tokens to highlight):';
                label.style.fontSize = '0.875rem';
                label.style.color = 'var(--text-secondary)';
                label.style.marginBottom = '8px';
                label.style.marginTop = '16px';
                
                // Add a note about the difference if texts don't match
                if (this.originalText !== this.reconstructedText) {
                    const note = document.createElement('div');
                    note.textContent = 'Note: This shows the text as reconstructed from tokens (may differ slightly from original)';
                    note.style.fontSize = '0.75rem';
                    note.style.color = 'var(--text-muted)';
                    note.style.marginBottom = '8px';
                    note.style.fontStyle = 'italic';
                    
                    visualTab.appendChild(separator);
                    visualTab.appendChild(label);
                    visualTab.appendChild(note);
                } else {
                    visualTab.appendChild(separator);
                    visualTab.appendChild(label);
                }
                
                highlightedTextContainer = document.createElement('div');
                highlightedTextContainer.id = 'highlightedText';
                highlightedTextContainer.className = 'highlighted-text';
                
                visualTab.appendChild(highlightedTextContainer);
                console.log('Created and appended highlighted text container');
            }
        }
        
        if (highlightedTextContainer) {
            // Create spans for each character that can be highlighted
            highlightedTextContainer.innerHTML = '';
            console.log('Creating character spans for text length:', this.reconstructedText.length);
            
            for (let i = 0; i < this.reconstructedText.length; i++) {
                const charSpan = document.createElement('span');
                charSpan.textContent = this.reconstructedText[i];
                charSpan.dataset.charIndex = i;
                charSpan.className = 'text-char';
                
                // Preserve whitespace and line breaks
                if (this.reconstructedText[i] === '\n') {
                    charSpan.innerHTML = '<br>';
                } else if (this.reconstructedText[i] === ' ') {
                    charSpan.innerHTML = '&nbsp;';
                }
                
                highlightedTextContainer.appendChild(charSpan);
            }
        }
    }
    
    highlightToken(index) {
        const visualToken = this.visualTokens.querySelector(`[data-token-index="${index}"]`);
        const tokenId = this.tokenIds.querySelector(`[data-token-index="${index}"]`);
        
        // Highlight the tokens
        if (visualToken) {
            visualToken.style.backgroundColor = '#58a6ff';
            visualToken.style.color = 'white';
            visualToken.style.transform = 'translateY(-1px)';
            visualToken.style.boxShadow = '0 2px 8px rgba(88, 166, 255, 0.3)';
            visualToken.style.zIndex = '10';
        }
        if (tokenId) {
            tokenId.style.backgroundColor = '#58a6ff';
            tokenId.style.color = 'white';
            tokenId.style.transform = 'translateY(-1px)';
            tokenId.style.boxShadow = '0 2px 8px rgba(88, 166, 255, 0.3)';
            tokenId.style.zIndex = '10';
        }
        
        // Highlight the corresponding text
        this.highlightTextForToken(index);
    }
    
    unhighlightToken(index) {
        const visualToken = this.visualTokens.querySelector(`[data-token-index="${index}"]`);
        const tokenId = this.tokenIds.querySelector(`[data-token-index="${index}"]`);
        
        // Remove highlight from tokens
        if (visualToken) {
            visualToken.style.backgroundColor = '';
            visualToken.style.color = '';
            visualToken.style.transform = '';
            visualToken.style.boxShadow = '';
            visualToken.style.zIndex = '';
        }
        if (tokenId) {
            tokenId.style.backgroundColor = '';
            tokenId.style.color = '';
            tokenId.style.transform = '';
            tokenId.style.boxShadow = '';
            tokenId.style.zIndex = '';
        }
        
        // Remove highlight from text
        this.unhighlightTextForToken(index);
    }
    
    highlightTextForToken(tokenIndex) {
        if (!this.tokenTextMapping || tokenIndex >= this.tokenTextMapping.length) return;
        
        const mapping = this.tokenTextMapping[tokenIndex];
        const highlightedTextContainer = document.getElementById('highlightedText');
        
        if (highlightedTextContainer && mapping.startPos !== -1 && mapping.endPos !== -1) {
            // Highlight characters in the text corresponding to this token
            for (let i = mapping.startPos; i < mapping.endPos; i++) {
                const charSpan = highlightedTextContainer.querySelector(`[data-char-index="${i}"]`);
                if (charSpan) {
                    charSpan.style.backgroundColor = '#58a6ff';
                    charSpan.style.color = 'white';
                    charSpan.style.borderRadius = '2px';
                    charSpan.style.padding = '1px 2px';
                }
            }
        }
    }
    
    unhighlightTextForToken(tokenIndex) {
        if (!this.tokenTextMapping || tokenIndex >= this.tokenTextMapping.length) return;
        
        const mapping = this.tokenTextMapping[tokenIndex];
        const highlightedTextContainer = document.getElementById('highlightedText');
        
        if (highlightedTextContainer && mapping.startPos !== -1 && mapping.endPos !== -1) {
            // Remove highlight from characters in the text
            for (let i = mapping.startPos; i < mapping.endPos; i++) {
                const charSpan = highlightedTextContainer.querySelector(`[data-char-index="${i}"]`);
                if (charSpan) {
                    charSpan.style.backgroundColor = '';
                    charSpan.style.color = '';
                    charSpan.style.borderRadius = '';
                    charSpan.style.padding = '';
                }
            }
        }
    }
    
    showMessage(message, type = 'info') {
        // Create a simple toast notification
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: 8px;
            padding: 12px 16px;
            color: var(--text-primary);
            z-index: 1000;
            max-width: 300px;
            box-shadow: 0 4px 12px var(--shadow-primary);
        `;
        
        // Add type-specific styling
        if (type === 'success') {
            toast.style.borderColor = 'var(--accent-success)';
        } else if (type === 'error') {
            toast.style.borderColor = 'var(--accent-danger)';
        } else if (type === 'warning') {
            toast.style.borderColor = 'var(--accent-warning)';
        }
        
        document.body.appendChild(toast);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 3000);
    }
    
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Modal functions (global scope for HTML onclick handlers)
function showAbout() {
    document.getElementById('aboutModal').style.display = 'block';
}

function closeAbout() {
    document.getElementById('aboutModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('aboutModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BPETokenizerApp();
});
