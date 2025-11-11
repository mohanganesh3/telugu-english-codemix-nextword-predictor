import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hmm.hmm_model import load_hmm_model
from models.ngram.ngram_model import load_ngram_model
from src.preprocessing.tokenizer import tokenize

def main():
    print("Loading models...")
    hmm_model = load_hmm_model()
    ngram_model = load_ngram_model()
    print("Models loaded. You can now enter text to get predictions.")

    while True:
        try:
            text = input("> ")
            if not text:
                continue

            tokens = tokenize(text)

            print("\nHMM Predictions:")
            hmm_preds = hmm_model.predict_next_word(tokens)
            for pred, score in hmm_preds:
                print(f"- {pred} (score: {score:.4f})")

            print("\nN-gram Predictions:")
            ngram_preds = ngram_model.predict_next_word(tokens)
            for pred, score in ngram_preds:
                print(f"- {pred} (score: {score:.4f})")
            
            print("-" * 20)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()

