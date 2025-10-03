# Video Caption Generation - Homework 2

This project implements a sequence-to-sequence model with attention for generating text captions from video features. I used the S2VT architecture, which processes video frames through an encoder LSTM and generates captions word-by-word through a decoder LSTM with an attention mechanism. 

## How to Run Everything

First install the required packages:
pip install torch numpy scipy

To train the model from scratch, run:
python3 model_seq2seq.py

To generate captions for test videos:
chmod +x hw2_seq2seq.sh
./hw2_seq2seq.sh MLDS_hw2_1_data/testing_data output.txt

To check the BLEU score:
python3 evaluate.py output.txt
