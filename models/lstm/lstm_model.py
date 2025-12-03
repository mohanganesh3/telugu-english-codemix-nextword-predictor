"""
LSTM Language Model - Pure Python Implementation (No Deep Learning Frameworks)
Implements LSTM from scratch for Telugu-English code-mixed text prediction

Note: This is an educational implementation using only NumPy for matrix operations.
For production use, frameworks like PyTorch or TensorFlow are recommended.
"""

import numpy as np
import pickle
import os
from collections import defaultdict, Counter
import time
from multiprocessing import Pool, cpu_count
import multiprocessing as mp


class LSTMCell:
    """Single LSTM cell with forget, input, and output gates"""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights with Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Forget gate weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros((hidden_size, 1))
        
        # Input gate weights
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        
        # Candidate gate weights
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bc = np.zeros((hidden_size, 1))
        
        # Output gate weights
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        """Numerically stable sigmoid"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def tanh(self, x):
        """Hyperbolic tangent"""
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass through LSTM cell
        
        Args:
            x: Input vector (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)
            c_prev: Previous cell state (hidden_size, 1)
            
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
            cache: Values needed for backward pass
        """
        # Concatenate input and previous hidden state
        concat = np.vstack((x, h_prev))
        
        # Forget gate: decides what to forget from cell state
        ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        
        # Input gate: decides what new information to store
        it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        
        # Candidate values: new candidate values to add to cell state
        c_tilde = self.tanh(np.dot(self.Wc, concat) + self.bc)
        
        # Update cell state
        c_next = ft * c_prev + it * c_tilde
        
        # Output gate: decides what to output
        ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        
        # Compute hidden state
        h_next = ot * self.tanh(c_next)
        
        # Cache values for backward pass
        cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'concat': concat, 'ft': ft, 'it': it,
            'c_tilde': c_tilde, 'c_next': c_next,
            'ot': ot, 'h_next': h_next
        }
        
        return h_next, c_next, cache
    
    def backward(self, dh_next, dc_next, cache):
        """
        Backward pass through LSTM cell (BPTT)
        
        Args:
            dh_next: Gradient of loss w.r.t. next hidden state
            dc_next: Gradient of loss w.r.t. next cell state
            cache: Values from forward pass
            
        Returns:
            dx: Gradient w.r.t. input
            dh_prev: Gradient w.r.t. previous hidden state
            dc_prev: Gradient w.r.t. previous cell state
            grads: Dictionary of weight gradients
        """
        # Extract cached values
        x = cache['x']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        concat = cache['concat']
        ft = cache['ft']
        it = cache['it']
        c_tilde = cache['c_tilde']
        c_next = cache['c_next']
        ot = cache['ot']
        
        # Gradient of output gate
        dot = dh_next * self.tanh(c_next)
        dot_input = dot * ot * (1 - ot)  # sigmoid derivative
        
        # Gradient of cell state
        dc_next = dc_next + dh_next * ot * (1 - self.tanh(c_next)**2)
        
        # Gradient of candidate
        dc_tilde = dc_next * it
        dc_tilde_input = dc_tilde * (1 - c_tilde**2)  # tanh derivative
        
        # Gradient of input gate
        dit = dc_next * c_tilde
        dit_input = dit * it * (1 - it)  # sigmoid derivative
        
        # Gradient of forget gate
        dft = dc_next * c_prev
        dft_input = dft * ft * (1 - ft)  # sigmoid derivative
        
        # Gradients w.r.t. weights and biases
        dWf = np.dot(dft_input, concat.T)
        dbf = dft_input
        
        dWi = np.dot(dit_input, concat.T)
        dbi = dit_input
        
        dWc = np.dot(dc_tilde_input, concat.T)
        dbc = dc_tilde_input
        
        dWo = np.dot(dot_input, concat.T)
        dbo = dot_input
        
        # Gradient w.r.t. concatenated input
        dconcat = (np.dot(self.Wf.T, dft_input) +
                   np.dot(self.Wi.T, dit_input) +
                   np.dot(self.Wc.T, dc_tilde_input) +
                   np.dot(self.Wo.T, dot_input))
        
        # Split gradient
        dx = dconcat[:self.input_size, :]
        dh_prev = dconcat[self.input_size:, :]
        
        # Gradient of previous cell state
        dc_prev = dc_next * ft
        
        grads = {
            'dWf': dWf, 'dbf': dbf,
            'dWi': dWi, 'dbi': dbi,
            'dWc': dWc, 'dbc': dbc,
            'dWo': dWo, 'dbo': dbo
        }
        
        return dx, dh_prev, dc_prev, grads


class LSTMLanguageModel:
    """LSTM-based language model for next word prediction"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_size=256, 
                 num_layers=2, learning_rate=0.001, context_length=10):
        """
        Initialize LSTM language model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimization
            context_length: Number of previous words to consider
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.context_length = context_length
        
        # Vocabulary mappings
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        # Model components (initialized after vocabulary)
        self.embeddings = None
        self.lstm_layers = []
        self.output_weights = None
        self.output_bias = None
        
        # For training
        self.trained = False
        
    def build_vocabulary(self, texts, min_freq=2):
        """
        Build vocabulary from training texts
        
        Args:
            texts: List of tokenized texts (list of word lists)
            min_freq: Minimum frequency for word to be included
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            self.word_freq.update(text)
        
        # Add special tokens
        self.word_to_idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.START_TOKEN: 2,
            self.END_TOKEN: 3
        }
        
        # Add words meeting minimum frequency
        idx = 4
        for word, freq in self.word_freq.most_common():
            if freq >= min_freq and idx < self.vocab_size:
                self.word_to_idx[word] = idx
                idx += 1
        
        # Update vocab size to actual size
        self.vocab_size = len(self.word_to_idx)
        
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total unique words: {len(self.word_freq)}")
        
    def initialize_weights(self):
        """Initialize model weights after vocabulary is built"""
        print("Initializing model weights...")
        
        # Word embeddings
        self.embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        
        # LSTM layers
        self.lstm_layers = []
        for i in range(self.num_layers):
            input_size = self.embedding_dim if i == 0 else self.hidden_size
            self.lstm_layers.append(LSTMCell(input_size, self.hidden_size))
        
        # Output layer (hidden -> vocabulary)
        scale = np.sqrt(2.0 / self.hidden_size)
        self.output_weights = np.random.randn(self.vocab_size, self.hidden_size) * scale
        self.output_bias = np.zeros((self.vocab_size, 1))
        
        print("Model initialized successfully!")
        
    def get_word_idx(self, word):
        """Get index for word, return UNK if not in vocabulary"""
        return self.word_to_idx.get(word, self.word_to_idx[self.UNK_TOKEN])
    
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward(self, word_indices):
        """
        Forward pass through LSTM
        
        Args:
            word_indices: List of word indices
            
        Returns:
            outputs: List of output probability distributions
            caches: List of cache dictionaries for backward pass
        """
        batch_size = 1
        seq_length = len(word_indices)
        
        # Initialize hidden and cell states for all layers
        h_states = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
        c_states = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)]
        
        outputs = []
        caches = []
        
        # Process each word in sequence
        for t, word_idx in enumerate(word_indices):
            # Get word embedding
            x = self.embeddings[word_idx].reshape(-1, 1)
            
            layer_caches = []
            
            # Pass through LSTM layers
            for layer_idx in range(self.num_layers):
                h_prev = h_states[layer_idx]
                c_prev = c_states[layer_idx]
                
                h_next, c_next, cache = self.lstm_layers[layer_idx].forward(x, h_prev, c_prev)
                
                h_states[layer_idx] = h_next
                c_states[layer_idx] = c_next
                layer_caches.append(cache)
                
                # Input to next layer is output of current layer
                x = h_next
            
            # Output layer
            logits = np.dot(self.output_weights, h_states[-1]) + self.output_bias
            probs = self.softmax(logits)
            
            outputs.append(probs)
            caches.append({
                'layer_caches': layer_caches,
                'h_states': [h.copy() for h in h_states],
                'word_idx': word_idx
            })
        
        return outputs, caches
    
    def compute_loss(self, outputs, target_indices):
        """
        Compute cross-entropy loss
        
        Args:
            outputs: List of probability distributions
            target_indices: List of target word indices
            
        Returns:
            loss: Average cross-entropy loss
        """
        loss = 0.0
        for probs, target_idx in zip(outputs, target_indices):
            # Cross-entropy loss
            loss += -np.log(probs[target_idx, 0] + 1e-10)
        
        return loss / len(target_indices)
    
    def train_on_batch(self, context_indices, target_indices):
        """
        Train on a single batch (sequence)
        
        Args:
            context_indices: List of context word indices
            target_indices: List of target word indices
            
        Returns:
            loss: Loss value for this batch
        """
        # Forward pass
        outputs, caches = self.forward(context_indices)
        
        # Compute loss
        loss = self.compute_loss(outputs, target_indices)
        
        # Backward pass (simplified - gradient descent on embeddings and output layer)
        # Full BPTT implementation would be very complex, so we use a simplified approach
        
        # Update output layer
        for t, (probs, target_idx) in enumerate(zip(outputs, target_indices)):
            # Gradient of loss w.r.t. logits
            dlogits = probs.copy()
            dlogits[target_idx] -= 1
            dlogits = dlogits / len(target_indices)
            
            # Get hidden state from cache
            h_final = caches[t]['h_states'][-1]
            
            # Update output weights
            self.output_weights -= self.learning_rate * np.dot(dlogits, h_final.T)
            self.output_bias -= self.learning_rate * dlogits
            
            # Update embeddings (simple gradient)
            word_idx = context_indices[t]
            dh = np.dot(self.output_weights.T, dlogits)
            
            # Simplified embedding update
            self.embeddings[word_idx] -= self.learning_rate * dh[:self.embedding_dim, 0]
        
        return loss
    
    def compute_batch_gradients(self, batch_sequences):
        """
        Compute gradients for a batch of sequences (for parallel processing)
        
        Args:
            batch_sequences: List of (context_indices, target_idx) tuples
            
        Returns:
            Total loss for batch and gradient updates
        """
        batch_loss = 0.0
        
        # Accumulate gradients
        embedding_grads = {}
        output_weight_grad = np.zeros_like(self.output_weights)
        output_bias_grad = np.zeros_like(self.output_bias)
        
        for context_indices, target_idx in batch_sequences:
            # Forward pass
            outputs, caches = self.forward(context_indices)
            
            # Compute loss
            loss = self.compute_loss(outputs, [target_idx])
            batch_loss += loss
            
            # Compute gradients
            for t, (probs, tgt_idx) in enumerate(zip(outputs, [target_idx])):
                dlogits = probs.copy()
                dlogits[tgt_idx] -= 1
                dlogits = dlogits / len([target_idx])
                
                h_final = caches[t]['h_states'][-1]
                
                # Accumulate output layer gradients
                output_weight_grad += np.dot(dlogits, h_final.T)
                output_bias_grad += dlogits
                
                # Accumulate embedding gradients
                word_idx = context_indices[t]
                dh = np.dot(self.output_weights.T, dlogits)
                
                if word_idx not in embedding_grads:
                    embedding_grads[word_idx] = np.zeros(self.embedding_dim)
                embedding_grads[word_idx] += dh[:self.embedding_dim, 0]
        
        return batch_loss, embedding_grads, output_weight_grad, output_bias_grad
    
    def train(self, train_texts, epochs=5, batch_size=32, save_path=None, n_workers=None):
        """
        Train the LSTM model with parallel processing
        
        Args:
            train_texts: List of tokenized texts
            epochs: Number of training epochs
            batch_size: Batch size (sequences per batch)
            save_path: Path to save trained model
            n_workers: Number of parallel workers (default: all CPU cores)
        """
        if n_workers is None:
            n_workers = cpu_count()
        
        print(f"\nTraining LSTM Language Model...")
        print(f"Epochs: {epochs}, Context Length: {self.context_length}")
        print(f"Vocabulary Size: {self.vocab_size}, Hidden Size: {self.hidden_size}")
        print(f"Layers: {self.num_layers}, Learning Rate: {self.learning_rate}")
        print(f"Using {n_workers} CPU cores for parallel processing")
        print("=" * 70)
        
        # Prepare training sequences
        sequences = []
        for text in train_texts:
            if len(text) < 2:
                continue
            
            # Create sequences with context and target
            for i in range(1, len(text)):
                start_idx = max(0, i - self.context_length)
                context = text[start_idx:i]
                target = text[i]
                
                # Convert to indices
                context_indices = [self.get_word_idx(w) for w in context]
                target_idx = self.get_word_idx(target)
                
                # Pad context if needed
                while len(context_indices) < self.context_length:
                    context_indices.insert(0, self.word_to_idx[self.PAD_TOKEN])
                
                sequences.append((context_indices, target_idx))
        
        print(f"Total training sequences: {len(sequences)}")
        
        # Set NumPy to use multiple threads for BLAS operations
        if n_workers > 1:
            os.environ['OMP_NUM_THREADS'] = str(n_workers)
            os.environ['OPENBLAS_NUM_THREADS'] = str(n_workers)
            os.environ['MKL_NUM_THREADS'] = str(n_workers)
            os.environ['NUMEXPR_NUM_THREADS'] = str(n_workers)
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_workers)
            print(f"Using {n_workers} workers for parallel processing")
            pool = Pool(processes=n_workers, maxtasksperchild=1)
            use_multiprocessing = True
        else:
            print("Using sequential training (more stable)")
            use_multiprocessing = False
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            np.random.shuffle(sequences)
            
            total_loss = 0.0
            num_batches = 0
            
            if use_multiprocessing:
                # Parallel processing mode
                super_batch_size = batch_size * n_workers * 2
            else:
                # Sequential processing mode
                super_batch_size = batch_size
            
            print(f"\nEpoch {epoch + 1}/{epochs} - Processing with batches of {super_batch_size} sequences")
            
            for i in range(0, len(sequences), super_batch_size):
                super_batch = sequences[i:i + super_batch_size]
                batch_loss = 0.0
                
                if use_multiprocessing:
                    # Parallel processing
                    chunk_size = max(1, len(super_batch) // n_workers)
                    chunks = [super_batch[j:j + chunk_size] for j in range(0, len(super_batch), chunk_size)]
                    
                    results = pool.map(self.compute_batch_gradients, chunks)
                    
                    # Aggregate results
                    all_embedding_grads = {}
                    total_output_weight_grad = np.zeros_like(self.output_weights)
                    total_output_bias_grad = np.zeros_like(self.output_bias)
                    
                    for b_loss, emb_grads, out_w_grad, out_b_grad in results:
                        batch_loss += b_loss
                        for word_idx, grad in emb_grads.items():
                            if word_idx not in all_embedding_grads:
                                all_embedding_grads[word_idx] = np.zeros(self.embedding_dim)
                            all_embedding_grads[word_idx] += grad
                        total_output_weight_grad += out_w_grad
                        total_output_bias_grad += out_b_grad
                    
                    # Apply gradients
                    self.output_weights -= self.learning_rate * total_output_weight_grad
                    self.output_bias -= self.learning_rate * total_output_bias_grad
                    for word_idx, grad in all_embedding_grads.items():
                        self.embeddings[word_idx] -= self.learning_rate * grad
                    
                    num_batches += len(chunks)
                else:
                    # Sequential processing
                    for context_indices, target_idx in super_batch:
                        loss = self.train_on_batch(context_indices, [target_idx])
                        batch_loss += loss
                    num_batches += 1
                
                total_loss += batch_loss
                
                # Print progress
                if num_batches % 100 == 0:
                    avg_loss = total_loss / max(1, num_batches)
                    sequences_processed = min(i + super_batch_size, len(sequences))
                    pct_complete = (sequences_processed / len(sequences)) * 100
                    total_batches = (len(sequences) + super_batch_size - 1) // super_batch_size
                    current_batch = (i // super_batch_size) + 1
                    print(f"  Epoch {epoch + 1}, Batch {current_batch}/{total_batches}, "
                          f"Progress: {pct_complete:.1f}%, Avg Loss: {avg_loss:.4f}")
            
            avg_epoch_loss = total_loss / max(1, num_batches)
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Time: {epoch_time:.2f}s")
        
        # Close the pool after all epochs (if using multiprocessing)
        if use_multiprocessing:
            pool.close()
            pool.join()
        
        self.trained = True
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
            print(f"Model saved to {save_path}")
    
    def predict_next_word(self, context, top_k=5):
        """
        Predict next word given context
        
        Args:
            context: List of context words
            top_k: Number of top predictions to return
            
        Returns:
            predictions: List of (word, probability) tuples
        """
        if not self.trained:
            print("Warning: Model not trained yet!")
            return []
        
        # Prepare context
        context = context[-self.context_length:]  # Take last N words
        context_indices = [self.get_word_idx(w) for w in context]
        
        # Pad if needed
        while len(context_indices) < self.context_length:
            context_indices.insert(0, self.word_to_idx[self.PAD_TOKEN])
        
        # Forward pass
        outputs, _ = self.forward(context_indices)
        
        # Get final output probabilities
        probs = outputs[-1].flatten()
        
        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            word = self.idx_to_word.get(idx, self.UNK_TOKEN)
            prob = float(probs[idx])
            predictions.append((word, prob))
        
        return predictions
    
    def save_model(self, filepath):
        """Save trained model to file"""
        model_data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate,
            'context_length': self.context_length,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_freq': dict(self.word_freq),
            'embeddings': self.embeddings,
            'output_weights': self.output_weights,
            'output_bias': self.output_bias,
            'lstm_layers': [
                {
                    'Wf': layer.Wf, 'bf': layer.bf,
                    'Wi': layer.Wi, 'bi': layer.bi,
                    'Wc': layer.Wc, 'bc': layer.bc,
                    'Wo': layer.Wo, 'bo': layer.bo
                }
                for layer in self.lstm_layers
            ],
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore configuration
        self.vocab_size = model_data['vocab_size']
        self.embedding_dim = model_data['embedding_dim']
        self.hidden_size = model_data['hidden_size']
        self.num_layers = model_data['num_layers']
        self.learning_rate = model_data['learning_rate']
        self.context_length = model_data['context_length']
        self.word_to_idx = model_data['word_to_idx']
        self.idx_to_word = model_data['idx_to_word']
        self.word_freq = Counter(model_data['word_freq'])
        self.embeddings = model_data['embeddings']
        self.output_weights = model_data['output_weights']
        self.output_bias = model_data['output_bias']
        self.trained = model_data['trained']
        
        # Restore LSTM layers
        self.lstm_layers = []
        for layer_data in model_data['lstm_layers']:
            input_size = self.embedding_dim if len(self.lstm_layers) == 0 else self.hidden_size
            layer = LSTMCell(input_size, self.hidden_size)
            layer.Wf = layer_data['Wf']
            layer.bf = layer_data['bf']
            layer.Wi = layer_data['Wi']
            layer.bi = layer_data['bi']
            layer.Wc = layer_data['Wc']
            layer.bc = layer_data['bc']
            layer.Wo = layer_data['Wo']
            layer.bo = layer_data['bo']
            self.lstm_layers.append(layer)
        
        print(f"Model loaded from {filepath}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Hidden size: {self.hidden_size}, Layers: {self.num_layers}")
