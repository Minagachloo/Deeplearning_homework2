import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import os
import json
import re
import pickle
import random
from scipy.special import expit


device_ = torch.device('cpu')
print(f"Using device: {device_}")


#DATA PREPROCESSING

def preprocessing_data(data_dir='data'):
    """Build vocabulary from training data"""
    file_path = os.path.join(data_dir, 'training_label.json')
    with open(file_path, 'r') as f:
        file_ = json.load(f)

    word_count_dict = {}
    for i in file_:
        for j in i['caption']:
            word_list = re.sub('[.!,;?]', ' ', j).split()
            for w in word_list:
                w = w.lower()
                word_count_dict[w] = word_count_dict.get(w, 0) + 1

    # Filter by min_count > 3 (baseline requirement)
    w_dict = {word: count for word, count in word_count_dict.items() if count > 3}
    
    # Special tokens
    tokens_ = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    
    index_to_word = {i + len(tokens_): w for i, w in enumerate(w_dict)}
    word_to_index = {w: i + len(tokens_) for i, w in enumerate(w_dict)}
    
    for t, i in tokens:
        index_to_word[i] = t
        word_to_index[t] = i

    print(f"Vocabulary size: {len(index_to_word)}")
    return index_to_word, word_to_index, w_dict


def s_split(sentence, w_dict, word_to_index):
    """Convert sentence to indices"""
    sentence_ = re.sub(r'[.!,;?]', ' ', sentence_).lower().split()
    indices = []
    for word in sentence_:
        if word in w_dict:
            indices.append(word_to_index[word])
        else:
            indices.append(3)  # <UNK>
    indices.insert(0, 1)  # <SOS>
    indices.append(2)      # <EOS>
    return indices


def annotate(label_file, w_dict, word_to_index, data_dir='data'):
    """Create (video_id, caption_indices) pairs"""
    label_json = os.path.join(data_dir, label_file)
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = s_split(s, w_dict, word_to_index)
            annotated_caption.append((d['id'], s))
    return annotated_caption


def avi(feat_dir, data_dir='data'):
    """Load all video features"""
    avi_data = {}
    training_feats = os.path.join(data_dir, feat_dir, 'feat')
    files = os.listdir(training_feats)
    
    for file in files:
        if file.endswith('.npy'):
            value = np.load(os.path.join(training_feats, file))
            avi_data[file.split('.npy')[0]] = value
    
    print(f"Loaded {len(avi_data)} video features")
    return avi_data


def batch(data_):
    """Custom collate function for DataLoader"""
    data_.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data_)
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths


#DATASET CLASSES 

class training_data(Dataset):
    def __init__(self, label_file, feat_dir, w_dict, word_to_index, data_dir='data'):
        self.label_file = label_file
        self.feat_dir = feat_dir
        self.w_dict = w_dict
        self.wordtoindex = word_to_index
        self.data_dir = data_dir
        
        # Load video features
        self.avi = avi(feat_dir, data_dir)
        
        # Create data pairs
        self.data_pair = annotate(label_file, w_dict, word_to_index, data_dir)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        avi_file_name, sentence_ = self.data_pair[idx]
        data_ = torch.Tensor(self.avi[avi_file_name])
        
        # Add small noise for regularization
        data_ += torch.Tensor(data_.size()).random_(0, 2000) / 10000.
        
        return data_, torch.LongTensor(sentence_)


class testing_dataset(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in sorted(files):
            if file.endswith('.npy'):
                key = file.split('.npy')[0]
                value = np.load(os.path.join(test_data_path, file))
                self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)

    def __getitem__(self, idx):
        return self.avi[idx]


#MODEL COMPONENTS

class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(2 * hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2 * self.hidden_size)

        x = self.l1(matching_inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        attention_weights = self.w(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context


class rnn_encoder(nn.Module):
    def __init__(self):
        super(rnn_encoder, self).__init__()
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, (hidden_state, cell_state) = self.lstm(input)
        return output, hidden_state


class rnn_decoder(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.3):
        super(rnn_decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, word_dim)
        self.dropout = nn.Dropout(dropout_percentage)
        self.lstm = nn.LSTM(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()

        decoder_current_hidden_state = encoder_last_hidden_state
        decoder_c = torch.zeros(decoder_current_hidden_state.size()).to(device)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long().to(device)
        
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len - 1):
            threshold = self.teacher_forcing_ratio(training_steps=tr_steps)
            
            if random.uniform(0.05, 0.995) > threshold:
                current_input_word = targets[:, i]
            else:
                current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, (decoder_current_hidden_state, decoder_c) = self.lstm(
                lstm_input, (decoder_current_hidden_state, decoder_c)
            )
            
            logprob = self.output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long().to(device)
        decoder_c = torch.zeros(decoder_current_hidden_state.size()).to(device)
        
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28

        for i in range(assumption_seq_len - 1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, (decoder_current_hidden_state, decoder_c) = self.lstm(
                lstm_input, (decoder_current_hidden_state, decoder_c)
            )
            
            logprob = self.output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def teacher_forcing_ratio(self, training_steps):
        return expit(training_steps / 20 + 0.85)


class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feat)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(
                encoder_last_hidden_state=encoder_last_hidden_state,
                encoder_output=encoder_outputs,
                targets=target_sentences, 
                mode=mode, 
                tr_steps=tr_steps
            )
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(
                encoder_last_hidden_state=encoder_last_hidden_state,
                encoder_output=encoder_outputs
            )
        return seq_logProb, seq_predictions


#TRAINING FUNCTIONS

def loss_cal(loss_fn, x, y, lengths):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        pre = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] - 1

        pre = pre[:seq_len]
        ground_truth = ground_truth[:seq_len]
        
        if flag:
            predict_cat = pre
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, pre), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)
    avg_loss = loss / batch_size

    return loss


def train(model, epoch, loss_fn, optimizer, train_loader):
    model.train()
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}")
    print(f"{'='*60}")
    epoch_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        avi_feats, ground_truths, lengths = batch
        avi_feats = avi_feats.to(device)
        ground_truths = ground_truths.to(device)

        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(
            avi_feats, 
            target_sentences=ground_truths, 
            mode='train', 
            tr_steps=epoch
        )
        
        ground_truths = ground_truths[:, 1:]  # Remove <SOS>
        loss = loss_cal(loss_fn, seq_logProb, ground_truths, lengths)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 5 == 0:
            print(f'Batch {batch_idx:3d}/{len(train_loader)} | Loss: {loss.item():.4f}')

    avg_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch} Average Loss: {avg_loss:.4f}")
    return avg_loss


def test(test_loader, model, indextoword):
    """Generate captions for test videos"""
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            id, avi_feats = batch
            avi_feats = torch.FloatTensor(avi_feats).to(device)

            seq_logProb, seq_predictions = model(avi_feats, mode='inference')
            
            # Convert predictions to words
            for i in range(len(id)):
                video_id = id[i]
                prediction = seq_predictions[i]
                
                sentence_ = []
                for idx in prediction:
                    word = indextoword[idx.item()]
                    if word == '<EOS>':
                        break
                    if word not in ['<PAD>', '<SOS>', '<UNK>']:
                        sentence_.append(word)
                    elif word == '<UNK>':
                        sentence_.append('something')
                
                results.append((video_id, ' '.join(sentence_)))
                
            if batch_idx % 10 == 0:
                print(f'Processed {batch_idx}/{len(test_loader)} batches')

    return results


def main():
    # Preprocessing
    print("="*60)
    print("Building vocabulary...")
    print("="*60)
    index_to_word, word_to_index, w_dict = preprocessing_data('data')
    
    # Save vocabulary
    os.makedirs('SavedModel', exist_ok=True)
    with open('SavedModel/index_to_word.pickle', 'wb') as handle:
        pickle.dump(index_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Vocabulary saved to SavedModel/index_to_word.pickle")
    
    # Create dataset
    print("\n" + "="*60)
    print("Creating dataset...")
    print("="*60)
    train_dataset = training_data(
        label_file='training_label.json',
        feat_dir='training_data',
        w_dict=w_dict,
        word_to_index=word_to_index,
        data_dir='data'
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        collate_fn=batch
    )

    # Model
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    vocab_size = len(index_to_word)
    print(f"Vocabulary size: {vocab_size}")
    
    encoder = rnn_encoder()
    decoder = rnn_decoder(512, vocab_size, vocab_size, 1024, 0.3)
    model = MODELS(encoder=encoder, decoder=decoder)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Training setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training
    epochs_n = 20 
    loss_arr = []
    
    print("\n" + "="*60)
    print("Starting Training")
    print(f"Total Epochs: {epochs_n}")
    print(f"Batch Size: 64")
    print(f"Learning Rate: 0.0001")
    print("="*60)
    
    for epoch in range(epochs_n):
        loss = train(model, epoch + 1, loss_fn, optimizer, train_dataloader)
        loss_arr.append(loss)
        
        # Save model every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"SavedModel/model_epoch_{epoch+1}.pth")
            print(f"\n>>> Model saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), "SavedModel/model_final.pth")
    torch.save(model, "SavedModel/complete_model.pth")  # Save complete model
    
    # Save loss history
    with open('SavedModel/loss_values.txt', 'w') as f:
        for item in loss_arr:
            f.write(f"{item}\n")
    
    print("\n" + "="*60)
    print("Training finished!")
    print(f"Final loss: {loss_arr[-1]:.4f}")
    print(f"Models saved in SavedModel/")
    print("="*60)


if __name__ == "__main__":
    main()
