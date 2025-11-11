"""
Run comprehensive evaluation with progress tracking
"""

from models.ngram.ngram_model import UltraAdvancedNgramModel  
from models.hmm.hmm_model import HMMModel
from models.hybrid.hybrid_language_tagger import HybridLanguageTagger
import time

print("=" * 80)
print("🔬 FINAL COMPREHENSIVE EVALUATION")
print("=" * 80)

# Load test data
print("\n📖 Loading test data...")
with open('data/processed/test.processed.txt', 'r', encoding='utf-8') as f:
    test_sentences = [line.strip().split() for line in f if line.strip()]

print(f"   Test sentences: {len(test_sentences):,}")

# Count predictions
total_predictions = sum(max(0, len(sent) - 1) for sent in test_sentences)
print(f"   Total predictions: {total_predictions:,}\n")

# Evaluate each model
results = {}

print("=" * 80)
print("1️⃣  EVALUATING N-GRAM MODEL")
print("=" * 80)

ngram = UltraAdvancedNgramModel(n=10)
ngram.load('models/ngram/ngram_model.txt')

start = time.time()
ngram_correct_top1 = 0
ngram_correct_top5 = 0
ngram_mrr = []
count = 0

for sent_idx, sent in enumerate(test_sentences):
    if len(sent) <= 1:
        continue
    
    for i in range(1, len(sent)):
        context = sent[max(0, i-10):i]
        true_word = sent[i]
        
        preds = ngram.predict_next_word(context, top_k=5)
        pred_words = [w for w, _ in preds]
        
        if pred_words and pred_words[0] == true_word:
            ngram_correct_top1 += 1
        
        if true_word in pred_words:
            ngram_correct_top5 += 1
            rank = pred_words.index(true_word) + 1
            ngram_mrr.append(1.0 / rank)
        else:
            ngram_mrr.append(0.0)
        
        count += 1
        
        if count % 5000 == 0:
            print(f"   {count:,} predictions | Top-1: {ngram_correct_top1/count*100:.2f}%")

ngram_time = time.time() - start
results['ngram'] = {
    'top1': ngram_correct_top1 / count * 100,
    'top5': ngram_correct_top5 / count * 100,
    'mrr': sum(ngram_mrr) / len(ngram_mrr),
    'time': ngram_time,
    'count': count
}

print(f"\n✅ N-gram: Top-1={results['ngram']['top1']:.2f}%, Top-5={results['ngram']['top5']:.2f}%, MRR={results['ngram']['mrr']:.4f}")
print(f"   Time: {ngram_time:.1f}s\n")

print("=" * 80)
print("2️⃣  EVALUATING HMM MODEL")
print("=" * 80)

hmm = HMMModel()
hmm.load_model('models/hmm_model.txt')

start = time.time()
hmm_correct_top1 = 0
hmm_correct_top5 = 0
hmm_mrr = []
count = 0

for sent_idx, sent in enumerate(test_sentences):
    if len(sent) <= 1:
        continue
    
    for i in range(1, len(sent)):
        context = sent[max(0, i-10):i]
        true_word = sent[i]
        
        preds = hmm.predict_next_word(context, top_k=5)
        pred_words = [w for w, _ in preds]
        
        if pred_words and pred_words[0] == true_word:
            hmm_correct_top1 += 1
        
        if true_word in pred_words:
            hmm_correct_top5 += 1
            rank = pred_words.index(true_word) + 1
            hmm_mrr.append(1.0 / rank)
        else:
            hmm_mrr.append(0.0)
        
        count += 1
        
        if count % 5000 == 0:
            print(f"   {count:,} predictions | Top-1: {hmm_correct_top1/count*100:.2f}%")

hmm_time = time.time() - start
results['hmm'] = {
    'top1': hmm_correct_top1 / count * 100,
    'top5': hmm_correct_top5 / count * 100,
    'mrr': sum(hmm_mrr) / len(hmm_mrr),
    'time': hmm_time,
    'count': count
}

print(f"\n✅ HMM: Top-1={results['hmm']['top1']:.2f}%, Top-5={results['hmm']['top5']:.2f}%, MRR={results['hmm']['mrr']:.4f}")
print(f"   Time: {hmm_time:.1f}s\n")

print("=" * 80)
print("3️⃣  EVALUATING ADVANCED HYBRID MODEL")
print("=" * 80)

hybrid = HybridLanguageTagger.load('models/hybrid_tagger.txt')

start = time.time()
hybrid_correct_top1 = 0
hybrid_correct_top5 = 0
hybrid_mrr = []
count = 0

for sent_idx, sent in enumerate(test_sentences):
    if len(sent) <= 1:
        continue
    
    for i in range(1, len(sent)):
        context = sent[max(0, i-10):i]
        true_word = sent[i]
        
        preds = hybrid.predict_next_word(context, top_k=5)
        pred_words = [w for w, _ in preds]
        
        if pred_words and pred_words[0] == true_word:
            hybrid_correct_top1 += 1
        
        if true_word in pred_words:
            hybrid_correct_top5 += 1
            rank = pred_words.index(true_word) + 1
            hybrid_mrr.append(1.0 / rank)
        else:
            hybrid_mrr.append(0.0)
        
        count += 1
        
        if count % 5000 == 0:
            print(f"   {count:,} predictions | Top-1: {hybrid_correct_top1/count*100:.2f}%")

hybrid_time = time.time() - start
results['hybrid'] = {
    'top1': hybrid_correct_top1 / count * 100,
    'top5': hybrid_correct_top5 / count * 100,
    'mrr': sum(hybrid_mrr) / len(hybrid_mrr),
    'time': hybrid_time,
    'count': count
}

print(f"\n✅ Hybrid: Top-1={results['hybrid']['top1']:.2f}%, Top-5={results['hybrid']['top5']:.2f}%, MRR={results['hybrid']['mrr']:.4f}")
print(f"   Time: {hybrid_time:.1f}s\n")

# Final comparison
print("=" * 80)
print("📊 FINAL RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Model':<30} {'Top-1':>10} {'Top-5':>10} {'MRR':>10} {'Time':>10}")
print("-" * 75)

for model_name, model_label in [('ngram', 'N-gram (10-gram)'), ('hmm', 'HMM (4-state)'), ('hybrid', 'Hybrid (Advanced)')]:
    r = results[model_name]
    print(f"{model_label:<30} {r['top1']:>9.2f}% {r['top5']:>9.2f}% {r['mrr']:>10.4f} {r['time']:>9.1f}s")

print("-" * 75)

# Best model
best_model = max(results.items(), key=lambda x: x[1]['top1'])
print(f"\n🏆 Best Model: {best_model[0].upper()}")
print(f"   Top-1 Accuracy: {best_model[1]['top1']:.2f}%")

# Improvement
improvement_top1 = results['hybrid']['top1'] - results['ngram']['top1']
improvement_top5 = results['hybrid']['top5'] - results['ngram']['top5']
improvement_mrr = results['hybrid']['mrr'] - results['ngram']['mrr']

print(f"\n📈 Hybrid vs N-gram:")
print(f"   Top-1: {improvement_top1:+.2f}% ({'+' if improvement_top1 > 0 else ''}{improvement_top1:.2f})")
print(f"   Top-5: {improvement_top5:+.2f}% ({'+' if improvement_top5 > 0 else ''}{improvement_top5:.2f})")
print(f"   MRR:   {improvement_mrr:+.4f} ({'+' if improvement_mrr > 0 else ''}{improvement_mrr:.4f})")

print("\n" + "=" * 80)
print("✅ EVALUATION COMPLETE!")
print("=" * 80)

# Save results
with open('final_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("FINAL EVALUATION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"{'Model':<30} {'Top-1':>10} {'Top-5':>10} {'MRR':>10} {'Time':>10}\n")
    f.write("-" * 75 + "\n")
    
    for model_name, model_label in [('ngram', 'N-gram (10-gram)'), ('hmm', 'HMM (4-state)'), ('hybrid', 'Hybrid (Advanced)')]:
        r = results[model_name]
        f.write(f"{model_label:<30} {r['top1']:>9.2f}% {r['top5']:>9.2f}% {r['mrr']:>10.4f} {r['time']:>9.1f}s\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"Best Model: {best_model[0].upper()}\n")
    f.write(f"Top-1 Accuracy: {best_model[1]['top1']:.2f}%\n")
    f.write("=" * 80 + "\n")

print("\n💾 Results saved to final_results.txt")
