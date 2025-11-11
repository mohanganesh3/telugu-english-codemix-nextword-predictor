import os
import sys
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.morphology.lemmatizer import lemmatize_word
from preprocessing.morphology.stop_words import STOP_WORDS
from preprocessing.normalizer import normalize_text
from preprocessing.tokenizer import tokenize


def preprocess_text(
    text: str,
    normalize: bool = True,
    remove_stop: bool = True,
    lemmatize: bool = True
) -> List[str]:

    if not text:
        return []

    # Step 1: Normalization
    if normalize:
        text = normalize_text(text)
    
    # Step 2: Tokenization
    tokens = tokenize(text)
    
    processed_tokens = []
    for token in tokens:
        # Step 3: Lemmatization
        if lemmatize:
            token = lemmatize_word(token)
            
        # Skip empty tokens
        if not token:
            continue
            
        # Skip stop words if requested
        if remove_stop and token.lower() in STOP_WORDS:
            continue
            
        processed_tokens.append(token)
    
    return processed_tokens


def preprocess_file(
    input_file: str,
    output_file: str,
    normalize: bool = True,
    remove_stop: bool = True,
    lemmatize: bool = True,
    skip_header: bool = True
) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_lines = 0
    processed_lines = 0

    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # Skip header if requested
        if skip_header:
            next(fin, None)
            
        # Write header to output
        fout.write("processed_text\n")
        
        for line in fin:
            total_lines += 1
            line = line.strip()
            
            if not line:
                continue
                
            try:
                processed = preprocess_text(
                    line,
                    normalize=normalize,
                    remove_stop=remove_stop,
                    lemmatize=lemmatize
                )
                if processed:
                    fout.write(' '.join(processed) + '\n')
                    processed_lines += 1
            except Exception as e:
                print(f"Warning: Error processing line {total_lines}: {str(e)}")
                continue
    
    print(f"\nPreprocessing Statistics:")
    print(f"Total lines read: {total_lines}")
    print(f"Lines processed: {processed_lines}")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Telugu-English code-mixed text.')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('--no-normalize', action='store_false', dest='normalize',
                      help='Disable text normalization')
    parser.add_argument('--no-stop', action='store_false', dest='remove_stop',
                      help='Disable stop word removal')
    parser.add_argument('--no-lemmatize', action='store_false', dest='lemmatize',
                      help='Disable lemmatization')
    parser.add_argument('--keep-header', action='store_false', dest='skip_header',
                      help='Keep the header row')
    
    args = parser.parse_args()
    
    try:
        preprocess_file(
            args.input_file,
            args.output_file,
            normalize=args.normalize,
            remove_stop=args.remove_stop,
            lemmatize=args.lemmatize,
            skip_header=args.skip_header
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)