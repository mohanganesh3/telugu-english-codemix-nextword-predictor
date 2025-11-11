class HMMSmoother:
    def __init__(self, model, method='laplace', alpha=1.0):
        self.model = model
        self.method = method
        self.alpha = alpha
    
    def laplace_smoothing(self, transition_probs, emission_probs):

        smoothed_transitions = {}
        for state_i in self.model.states:
            total = sum(transition_probs.get((state_i, state_j), 0) 
                       for state_j in self.model.states)
            for state_j in self.model.states:
                count = transition_probs.get((state_i, state_j), 0)
                smoothed_transitions[(state_i, state_j)] = \
                    (count + self.alpha) / (total + self.alpha * len(self.model.states))
        
        # Smooth emission probabilities
        smoothed_emissions = {}
        vocab_size = len(self.model.vocabulary)
        for state in self.model.states:
            total = sum(emission_probs.get((state, word), 0) 
                       for word in self.model.vocabulary)
            for word in self.model.vocabulary:
                count = emission_probs.get((state, word), 0)
                smoothed_emissions[(state, word)] = \
                    (count + self.alpha) / (total + self.alpha * vocab_size)
        
        return smoothed_transitions, smoothed_emissions
    
    def interpolation_smoothing(self, transition_probs, emission_probs, lambda_param=0.9):
        # Smooth transition probabilities
        uniform_trans_prob = 1.0 / len(self.model.states)
        smoothed_transitions = {}
        for state_i in self.model.states:
            total = sum(transition_probs.get((state_i, state_j), 0) 
                       for state_j in self.model.states)
            for state_j in self.model.states:
                if total > 0:
                    mle_prob = transition_probs.get((state_i, state_j), 0) / total
                else:
                    mle_prob = 0
                smoothed_transitions[(state_i, state_j)] = \
                    lambda_param * mle_prob + (1 - lambda_param) * uniform_trans_prob
        
        # Smooth emission probabilities
        uniform_emit_prob = 1.0 / len(self.model.vocabulary)
        smoothed_emissions = {}
        for state in self.model.states:
            total = sum(emission_probs.get((state, word), 0) 
                       for word in self.model.vocabulary)
            for word in self.model.vocabulary:
                if total > 0:
                    mle_prob = emission_probs.get((state, word), 0) / total
                else:
                    mle_prob = 0
                smoothed_emissions[(state, word)] = \
                    lambda_param * mle_prob + (1 - lambda_param) * uniform_emit_prob
        
        return smoothed_transitions, smoothed_emissions
    
    def smooth_probabilities(self, transition_probs, emission_probs):
        if self.method == 'laplace':
            return self.laplace_smoothing(transition_probs, emission_probs)
        elif self.method == 'interpolation':
            return self.interpolation_smoothing(transition_probs, emission_probs)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")