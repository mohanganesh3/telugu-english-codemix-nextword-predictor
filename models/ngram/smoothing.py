class NgramSmoother:
    def __init__(self, model, method='laplace', alpha=0.4, delta=0.75):
        self.model = model
        self.method = method
        self.alpha = alpha  # for stupid backoff
        self.delta = delta  # for kneser ney
        
        if method == 'kneser_ney':
            self.continuation_counts = self._compute_continuation_counts()
    
    def _compute_continuation_counts(self):
        counts = {}
        for ngram in self.model.ngram_counts:
            if len(ngram) > 1:
                word = ngram[-1]
                counts[word] = counts.get(word, 0) + 1
        return counts
    
    def laplace_smoothing(self, word, context):
        if not context:
            return (self.model.ngram_counts.get((word,), 0) + 1) / (sum(self.model.ngram_counts.values()) + len(self.model.vocabulary))
        
        context = tuple(context)
        ngram = context + (word,)
        
        numerator = self.model.ngram_counts.get(ngram, 0) + 1
        denominator = self.model.context_counts.get(context, 0) + len(self.model.vocabulary)
        
        return numerator / denominator
    
    def stupid_backoff(self, word, context):
        if not context:
            return self.model.ngram_counts.get((word,), 0) / sum(self.model.ngram_counts.values())
        
        context = tuple(context)
        ngram = context + (word,)
        
        # Try full n-gram
        if ngram in self.model.ngram_counts:
            return self.model.ngram_counts[ngram] / self.model.context_counts[context]
        
        # Backoff to lower order with penalty
        return self.alpha * self.stupid_backoff(word, context[1:])
    
    def kneser_ney(self, word, context):
        if not context:
            # For unigrams, use continuation counts
            return self.continuation_counts.get(word, 0) / sum(self.continuation_counts.values())
        
        context = tuple(context)
        ngram = context + (word,)
        
        # Get counts
        count = self.model.ngram_counts.get(ngram, 0)
        context_count = self.model.context_counts.get(context, 0)
        
        if context_count == 0:
            return self.kneser_ney(word, context[1:])
        
        # Apply smoothing
        lambda_factor = self.delta * len([n for n in self.model.ngram_counts if n[:-1] == context]) / context_count
        return max(count - self.delta, 0) / context_count + lambda_factor * self.kneser_ney(word, context[1:])
    
    def get_probability(self, word, context):
        if self.method == 'laplace':
            return self.laplace_smoothing(word, context)
        elif self.method == 'stupid_backoff':
            return self.stupid_backoff(word, context)
        else:  # kneser_ney
            return self.kneser_ney(word, context)