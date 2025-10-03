# test_model.py (corrected version)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import os
import pickle

from model_seq2seq import (
    attention, rnn_encoder, rnn_decoder, MODELS,
    testing_dataset
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(test_loader, model, indextoword):
    model.eval()
    results = []

    print(f"Generating captions for {len(test_loader)} videos...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            id, avi_feats = batch
            # FIX: Convert to float32 properly
            avi_feats = torch.from_numpy(np.array(avi_feats)).float().to(device)

            seq_logProb, seq_predictions = model(avi_feats, mode='inference')
            
            for i in range(len(id)):
                video_id = id[i]
                prediction = seq_predictions[i]
                
                sentence = []
                for idx in prediction:
                    word = indextoword[idx.item()]
                    if word == '<EOS>':
                        break
                    if word not in ['<PAD>', '<SOS>', '<UNK>']:
                        sentence.append(word)
                    elif word == '<UNK>':
                        sentence.append('something')
                
                results.append((video_id, ' '.join(sentence)))
                
            if (batch_idx + 1) % 10 == 0:
                print(f'Processed {batch_idx + 1}/{len(test_loader)} batches')

    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_model.py <test_data_dir> <output_file>")
        sys.exit(1)
    
    test_data_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    print("="*60)
    print("Loading model and vocabulary...")
    print("="*60)
    
    with open('SavedModel/indextoword.pickle', 'rb') as handle:
        indextoword = pickle.load(handle)
    
    vocab_size = len(indextoword)
    print(f"Vocabulary size: {vocab_size}")
    
    encoder = rnn_encoder()
    decoder = rnn_decoder(512, vocab_size, vocab_size, 1024, 0.3)
    model = MODELS(encoder=encoder, decoder=decoder)
    
    model.load_state_dict(torch.load('SavedModel/model_final.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    test_feat_dir = os.path.join(test_data_dir, 'feat')
    test_dataset = testing_dataset(test_feat_dir)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    print(f"Found {len(test_dataset)} test videos")
    
    results = test(test_dataloader, model, indextoword)
    
    with open(output_file, 'w') as f:
        for video_id, caption in results:
            f.write(f"{video_id},{caption}\n")
    
    print(f"Results saved to {output_file}")
    print(f"Total videos processed: {len(results)}")

if __name__ == "__main__":
    main()
