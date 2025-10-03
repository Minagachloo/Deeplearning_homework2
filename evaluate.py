import sys
import os

# Add data directory to path to import bleu_eval
sys.path.insert(0, 'data')

try:
    from bleu_eval import BLEU
except ImportError:
    print("Error: Could not import bleu_eval.py from data/")
    print("Make sure bleu_eval.py exists in data/ directory")
    sys.exit(1)

import json


def evaluate(output_file, reference_file='data/testing_label.json'):
    """Evaluate BLEU scores for generated captions"""
    
    # Load generated captions
    result = {}
    with open(output_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split(',', 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    
    print(f"Loaded {len(result)} generated captions from {output_file}")
    
    # Load ground truth
    with open(reference_file, 'r') as f:
        test = json.load(f)
    
    print(f"Loaded {len(test)} reference videos from {reference_file}")
    
    # Calculate BLEU scores
    bleu_scores = []
    for item in test:
        video_id = item['id']
        if video_id not in result:
            print(f"Warning: No caption found for video {video_id}")
            continue
            
        captions = [x.rstrip('.') for x in item['caption']]
        score = BLEU(result[video_id], captions, True)
        bleu_scores.append(score)
    
    # Calculate average
    if len(bleu_scores) == 0:
        print("Error: No valid BLEU scores calculated")
        sys.exit(1)
    
    average = sum(bleu_scores) / len(bleu_scores)
    
    print("\n" + "="*60)
    print("BLEU Evaluation Results")
    print("="*60)
    print(f"Average BLEU@1 score: {average:.4f}")
    print(f"Videos evaluated: {len(bleu_scores)}/{len(test)}")
    print("="*60)
    
    if average >= 0.6:
        print("\n✓ Baseline achieved! (BLEU@1 ≥ 0.6)")
    else:
        print(f"\n✗ Below baseline (need 0.6, got {average:.4f})")
        print("  Consider training for more epochs to improve score")
    
    return average


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <output_file>")
        print("Example: python evaluate.py output.txt")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    if not os.path.exists(output_file):
        print(f"Error: Output file '{output_file}' not found")
        sys.exit(1)
    
    evaluate(output_file)
