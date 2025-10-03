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
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cpu')
print(f"Using device: {device}")


# DATA PREPROCESSING

def build_vocab(data_dir='data'):
    """Build vocabulary from training data"""
    path = os.path.join(data_dir, 'training_label.json')
    with open(path, 'r') as f:
        data = json.load(f)

    word_freq = {}
    for item in data:
        for cap in item['caption']:
            words = re.sub('[.!,;?]', ' ', cap).split()
            for w in words:
                w = w.lower()
                word_freq[w] = word_freq.get(w, 0) + 1

    vocab = {w: c for w, c in word_freq.items() if c > 3}
    
    special = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    
    i2w = {i + len(special): w for i, w in enumerate(vocab)}
    w2i = {w: i + len(special) for i, w in enumerate(vocab)}
    
    for tok, idx in special:
        i2w[idx] = tok
        w2i[tok] = idx

    print(f"Vocabulary size: {len(i2w)}")
    return i2w, w2i, vocab


def sent_to_idx(sent, vocab, w2i):
    """Convert sentence to indices"""
    sent = re.sub(r'[.!,;?]', ' ', sent).lower().split()
    ids = []
    for w in sent:
        if w in vocab:
            ids.append(w2i[w])
        else:
            ids.append(3)
    ids.insert(0, 1)
    ids.append(2)
    return ids


def load_caps(label_file, vocab, w2i, data_dir='data'):
    """Load captions"""
    path = os.path.join(data_dir, label_file)
    pairs = []
    with open(path, 'r') as f:
        data = json.load(f)
    for item in data:
        for cap in item['caption']:
            ids = sent_to_idx(cap, vocab, w2i)
            pairs.append((item['id'], ids))
    return pairs


def load_feats(feat_dir, data_dir='data'):
    """Load video features"""
    feats = {}
    path = os.path.join(data_dir, feat_dir, 'feat')
    files = os.listdir(path)
    
    for f in files:
        if f.endswith('.npy'):
            vid = f.split('.npy')[0]
            feat = np.load(os.path.join(path, f))
            feats[vid] = feat
    
    print(f"Loaded {len(feats)} video features")
    return feats


def collate(batch):
    """Collate batch"""
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    vids, caps = zip(*batch)
    vids = torch.stack(vids, 0)

    lens = [len(c) for c in caps]
    padded = torch.zeros(len(caps), max(lens)).long()
    for i, c in enumerate(caps):
        padded[i, :lens[i]] = c[:lens[i]]
    return vids, padded, lens


# DATASET

class TrainData(Dataset):
    def __init__(self, label_file, feat_dir, vocab, w2i, data_dir='data'):
        self.vocab = vocab
        self.w2i = w2i
        self.feats = load_feats(feat_dir, data_dir)
        self.pairs = load_caps(label_file, vocab, w2i, data_dir)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vid, cap = self.pairs[idx]
        feat = torch.Tensor(self.feats[vid])
        feat += torch.Tensor(feat.size()).random_(0, 2000) / 10000.
        return feat, torch.LongTensor(cap)


class TestData(Dataset):
    def __init__(self, path):
        self.data = []
        files = os.listdir(path)
        for f in sorted(files):
            if f.endswith('.npy'):
                vid = f.split('.npy')[0]
                feat = np.load(os.path.join(path, f))
                self.data.append([vid, feat])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# MODEL

class Attention(nn.Module):
    def __init__(self, hidden):
        super(Attention, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(2 * hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, h, enc_out):
        bs, seq, feat = enc_out.size()
        h = h.view(bs, 1, feat).repeat(1, seq, 1)
        combined = torch.cat((enc_out, h), 2).view(-1, 2 * self.hidden)

        x = self.fc1(combined)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        attn = self.score(x)
        attn = attn.view(bs, seq)
        attn = F.softmax(attn, dim=1)
        
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(4096, 512)
        self.drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, x):
        bs, seq, feat = x.size()
        x = x.view(-1, feat)
        x = self.fc(x)
        x = self.drop(x)
        x = x.view(bs, seq, 512)
        out, (h, c) = self.lstm(x)
        return out, h


class Decoder(nn.Module):
    def __init__(self, hidden, vocab_size, word_dim, drop=0.3):
        super(Decoder, self).__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embed = nn.Embedding(vocab_size, word_dim)
        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(hidden + word_dim, hidden, batch_first=True)
        self.attn = Attention(hidden)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, h, enc_out, tgt=None, mode='train', epoch=None):
        _, bs, _ = h.size()
        c = torch.zeros(h.size()).to(device)
        word = Variable(torch.ones(bs, 1)).long().to(device)
        
        preds = []
        logits_list = []

        tgt = self.embed(tgt)
        _, max_len, _ = tgt.size()

        for i in range(max_len - 1):
            thresh = self.teach_ratio(epoch)
            
            if random.uniform(0.05, 0.995) > thresh:
                inp = tgt[:, i]
            else:
                inp = self.embed(word).squeeze(1)

            ctx = self.attn(h, enc_out)
            lstm_in = torch.cat([inp, ctx], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            
            logits = self.fc(out.squeeze(1))
            logits_list.append(logits.unsqueeze(1))
            word = logits.unsqueeze(1).max(2)[1]

        logits_list = torch.cat(logits_list, dim=1)
        preds = logits_list.max(2)[1]
        return logits_list, preds

    def infer(self, h, enc_out):
        _, bs, _ = h.size()
        c = torch.zeros(h.size()).to(device)
        word = Variable(torch.ones(bs, 1)).long().to(device)
        
        preds = []
        logits_list = []
        max_len = 28

        for i in range(max_len - 1):
            inp = self.embed(word).squeeze(1)
            ctx = self.attn(h, enc_out)
            lstm_in = torch.cat([inp, ctx], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            
            logits = self.fc(out.squeeze(1))
            logits_list.append(logits.unsqueeze(1))
            word = logits.unsqueeze(1).max(2)[1]

        logits_list = torch.cat(logits_list, dim=1)
        preds = logits_list.max(2)[1]
        return logits_list, preds

    def teach_ratio(self, epoch):
        return expit(epoch / 20 + 0.85)


class Model(nn.Module):
    def __init__(self, enc, dec):
        super(Model, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, vid, mode, tgt=None, epoch=None):
        enc_out, h = self.enc(vid)
        if mode == 'train':
            logits, preds = self.dec(h, enc_out, tgt, mode, epoch)
        else:
            logits, preds = self.dec.infer(h, enc_out)
        return logits, preds


# TRAIN

def calc_loss(criterion, pred, gt, lens):
    bs = len(pred)
    all_pred = []
    all_gt = []

    for i in range(bs):
        p = pred[i]
        g = gt[i]
        l = lens[i] - 1
        all_pred.append(p[:l])
        all_gt.append(g[:l])
    
    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    
    loss = criterion(all_pred, all_gt)
    return loss


def train_epoch(model, ep, criterion, opt, loader):
    model.train()
    print(f"\n{'='*60}")
    print(f"Epoch {ep + 1}")
    print(f"{'='*60}")
    total = 0

    for i, (vids, caps, lens) in enumerate(loader):
        vids = vids.to(device)
        caps = caps.to(device)

        opt.zero_grad()
        logits, preds = model(vids, mode='train', tgt=caps, epoch=ep)
        
        caps = caps[:, 1:]
        loss = calc_loss(criterion, logits, caps, lens)
        
        loss.backward()
        opt.step()
        total += loss.item()
        
        if i % 5 == 0:
            print(f'Batch {i:3d}/{len(loader)} | Loss: {loss.item():.4f}')

    avg = total / len(loader)
    print(f"\nEpoch {ep + 1} Average Loss: {avg:.4f}")
    return avg


def get_samples(model, loader, i2w, ep, n=10):
    model.eval()
    samps = []
    
    with torch.no_grad():
        for i, (vids, caps, lens) in enumerate(loader):
            if i >= n:
                break
            vids = vids.to(device)
            logits, preds = model(vids, mode='inference')
            
            gt = []
            for idx in caps[0]:
                w = i2w[idx.item()]
                if w == '<EOS>':
                    break
                if w not in ['<PAD>', '<SOS>']:
                    gt.append(w)
            
            pred = []
            for idx in preds[0]:
                w = i2w[idx.item()]
                if w == '<EOS>':
                    break
                if w not in ['<PAD>', '<SOS>', '<UNK>']:
                    pred.append(w)
            
            samps.append({'gt': ' '.join(gt), 'pred': ' '.join(pred)})
    
    model.train()
    return samps


def test(loader, model, i2w):
    model.eval()
    results = []

    with torch.no_grad():
        for vids, feats in loader:
            feats = torch.FloatTensor(feats).to(device)
            logits, preds = model(feats, mode='inference')
            
            for i in range(len(vids)):
                vid = vids[i]
                pred = preds[i]
                
                words = []
                for idx in pred:
                    w = i2w[idx.item()]
                    if w == '<EOS>':
                        break
                    if w not in ['<PAD>', '<SOS>', '<UNK>']:
                        words.append(w)
                    elif w == '<UNK>':
                        words.append('something')
                
                results.append((vid, ' '.join(words)))

    return results


def plot_loss(losses, path='SavedModel/loss_plot.png'):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Loss plot saved: {path}")


def plot_loss_vals(losses, path='SavedModel/loss_values.png'):
    """Plot loss values"""
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, 'bo-', linewidth=2, markersize=8)
    
    for i, loss in enumerate(losses):
        plt.text(i+1, loss, f'{loss:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Values per Epoch', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Loss values plot saved: {path}")


def save_data_info(vocab, i2w, path='SavedModel'):
    """Save vocabulary and data info"""
    with open(f'{path}/vocab_info.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATA PREPARATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Vocabulary Size: {len(i2w)}\n")
        f.write(f"Min Word Count: > 3\n\n")
        f.write("Special Tokens:\n")
        f.write("  <PAD>: 0 (padding)\n")
        f.write("  <SOS>: 1 (start of sentence)\n")
        f.write("  <EOS>: 2 (end of sentence)\n")
        f.write("  <UNK>: 3 (unknown word)\n\n")
        f.write("="*60 + "\n")
        f.write("TOP 20 WORDS\n")
        f.write("="*60 + "\n")
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:20]
        for idx, (word, count) in enumerate(sorted_vocab, 1):
            f.write(f"{idx:2d}. {word:15s} : {count:4d}\n")
    print(f"Vocabulary info saved: {path}/vocab_info.txt")


def main():
    start = time.time()
    
    print("="*60)
    print("Building vocabulary...")
    print("="*60)
    i2w, w2i, vocab = build_vocab('data')
    
    os.makedirs('SavedModel', exist_ok=True)
    with open('SavedModel/i2w.pickle', 'wb') as f:
        pickle.dump(i2w, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    save_data_info(vocab, i2w)
    
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    train_data = TrainData('training_label.json', 'training_data', vocab, w2i, 'data')
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(train_data, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate)

    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    vocab_size = len(i2w)
    print(f"Vocabulary size: {vocab_size}")
    
    enc = Encoder()
    dec = Decoder(512, vocab_size, 1024, 0.3)
    model = Model(enc, dec)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.0001)
    
    n_epochs = 20
    losses = []
    
    print("\n" + "="*60)
    print("Starting Training")
    print(f"Total Epochs: {n_epochs}")
    print(f"Batch Size: 64")
    print(f"Learning Rate: 0.0001")
    print("="*60)
    
    os.makedirs('outputs', exist_ok=True)
    log = open('outputs/training_log.txt', 'w')
    
    for ep in range(n_epochs):
        ep_start = time.time()
        loss = train_epoch(model, ep, criterion, opt, train_loader)
        losses.append(loss)
        ep_time = time.time() - ep_start
        
        if ep in [0, 1, 4, 9, 19]:
            print("\n" + "-"*60)
            print("Output Caption:")
            print("-"*60)
            samps = get_samples(model, val_loader, i2w, ep)
            for idx, s in enumerate(samps, 1):
                print(f"{idx}. {s['pred']}")
            
            log.write(f"\n{'='*50}\n")
            log.write(f"Epoch: {ep + 1}, Loss: {loss:.6f}, Time: {ep_time:.2f}s\n")
            log.write(f"{'='*50}\n")
            for idx, s in enumerate(samps):
                log.write(f"Sample {idx+1}:\n")
                log.write(f"GT:   {s['gt']}\n")
                log.write(f"Pred: {s['pred']}\n\n")
        
        if (ep + 1) % 20 == 0:
            torch.save(model.state_dict(), f"SavedModel/model_epoch_{ep+1}.pth")
            print(f"\n>>> Model saved at epoch {ep+1}")

    torch.save(model.state_dict(), "SavedModel/model_final.pth")
    torch.save(model, "SavedModel/model_complete.pth")
    
    with open('SavedModel/loss_values.txt', 'w') as f:
        for l in losses:
            f.write(f"{l}\n")
    
    plot_loss(losses)
    plot_loss_vals(losses)
    
    total = time.time() - start
    
    print("\n" + "="*60)
    print("TRAINING FINISHED!")
    print("="*60)
    print(f"Total time cost: {total:.2f} seconds ~ {total/3600:.2f} hours")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Models saved in SavedModel/")
    print("="*60)
    
    log.write(f"\n{'='*50}\n")
    log.write(f"Total time cost: {total:.2f} seconds ~ {total/3600:.2f} hours\n")
    log.write(f"{'='*50}\n")
    log.close()


if __name__ == "__main__":
    main()
