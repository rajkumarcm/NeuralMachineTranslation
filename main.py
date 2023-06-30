import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
from torchtext.data import get_tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import *
from torch.nn import functional as F
from tqdm import tqdm
import re

class NMTDataset(Dataset):
    def __init__(self, df, tokenizer, en_vocab, fr_vocab,
                 max_input_len, max_output_len, en_vocab_size, fr_vocab_size):
        super(NMTDataset, self).__init__()
        self.df = df
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.en_vocab_size = en_vocab_size
        self.fr_vocab_size = fr_vocab_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        english = self.df.iloc[idx]['english']
        en_tokens = word_tokenize(english)
        en_tokens = list(map(self.en_vocab.get, en_tokens))
        en_tokens = torch.Tensor(en_tokens).to(torch.int64)[None, ...]
        # en_tokens = F.one_hot(torch.Tensor(en_tokens).to(torch.int64), num_classes=self.en_vocab_size)[None, ...]
        en_tokens = pad_packed_sequence(pack_sequence(en_tokens), batch_first=True, total_length=self.max_input_len)[0]

        french = self.df.iloc[idx]['french']
        fr_tokens = word_tokenize(french)
        fr_tokens = ["<SOS>"] + fr_tokens + ["<EOS>"]
        fr_tokens = list(map(self.fr_vocab.get, fr_tokens))
        fr_tokens = torch.Tensor(fr_tokens).to(torch.int64)[None, ...]
        # fr_tokens = F.one_hot(torch.Tensor(fr_tokens).to(torch.int64), num_classes=self.fr_vocab_size)[None, ...]
        fr_tokens = pad_packed_sequence(pack_sequence(fr_tokens), batch_first=True, total_length=self.max_input_len)[0]
        return en_tokens, fr_tokens


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, masked, input_dim, d_embed, n_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.masked = masked
        self.input_dim = input_dim
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_model = d_model
        self.masked = masked
        self.softmax = torch.nn.Softmax()
        self.linear = torch.nn.Linear(d_embed * n_heads, d_model)

        self.L_q = torch.nn.Linear(input_dim, d_embed * n_heads)
        self.L_k = torch.nn.Linear(input_dim, d_embed * n_heads)
        self.L_v = torch.nn.Linear(input_dim, d_embed * n_heads)


    def scaled_dot_product_attention(self, Q, K, V, mask=False):
        d_k = K.shape[0]
        sim_score = torch.matmul(Q, K.transpose(-1, -2))/(np.sqrt(d_k))
        if mask:
            sim_score = torch.tril(sim_score, diagonal=0)
            sim_score[sim_score == 0] = -1e9
        sim_score = self.softmax(sim_score)
        return torch.matmul(sim_score, V)

    def split_heads(self, X):
        batch_size, n_words, _ = X.size()
        return X.view(batch_size, n_words, self.n_heads, self.d_embed).transpose(1, 2)

    def combine_heads(self, X):
        # batch_size x n_heads x n_words x embedding_size
        batch_size, _, n_words, _ = X.size()
        return X.transpose(1, 2).contiguous().view(batch_size, n_words, self.d_embed * self.n_heads)

    def forward(self, X1, X2):
        Q = self.L_q(X1)
        K = self.L_k(X2)
        V = self.L_v(X2)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        outputs = self.scaled_dot_product_attention(Q, K, V, self.masked)
        outputs = self.combine_heads(outputs)
        return self.linear(outputs)


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.act = torch.nn.ReLU()

    def forward(self, X):
        return self.linear2(self.act(self.linear1(X)))


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.Tensor([10000.0])) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, d_embed, n_heads, d_model):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_model = d_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mhead_attention = MultiHeadAttention(False, d_embed, d_embed, n_heads, d_model).to(device)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.linear = PositionWiseFeedForward(d_model, d_model).to(device)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.act = torch.nn.ReLU()
        # self.L_q = torch.nn.Linear(input_dim, d_embed * n_heads)
        # self.L_k = torch.nn.Linear(input_dim, d_embed * n_heads)
        # self.L_v = torch.nn.Linear(input_dim, d_embed * n_heads)

    def forward(self, X):
        # Q = self.L_q(X)
        # K = self.L_k(X)
        # V = self.L_v(X)

        Y = self.mhead_attention(X, X)
        Y = self.ln1(X + Y)
        Y = self.act(self.ln2(Y + self.linear(Y)))
        return Y


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, d_embed, n_heads, d_model):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_model = d_model
        self.act = torch.nn.ReLU()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mhead_attention1 = MultiHeadAttention(True, d_embed, d_embed, n_heads, d_model).to(device)
        self.ln1 = torch.nn.LayerNorm(d_embed)
        self.mhead_attention2 = MultiHeadAttention(False, d_embed, d_embed, n_heads, d_model).to(device)
        self.ln2 = torch.nn.LayerNorm(d_embed)
        self.linear = PositionWiseFeedForward(d_embed, d_model).to(device)
        self.ln3 = torch.nn.LayerNorm(d_model)


    def forward(self, encoder_output, D_X):
        outputs = self.mhead_attention1(D_X, D_X)
        D_Q = self.ln1(D_X + outputs)
        outputs = self.mhead_attention2(D_Q, encoder_output)
        D_V = self.act(self.ln2(D_Q + outputs))
        outputs = self.act(self.ln2(D_V + self.linear(D_V)))
        return outputs


class Transformer(torch.nn.Module):
    def __init__(self, n_heads, n_layers, d_model, vocab_size, fr_vocab_size):
        super(Transformer, self).__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_embed = d_model  # for now
        self.vocab_size = vocab_size
        self.fr_vocab_size = vocab_size
        self.input_embedding = torch.nn.Embedding(vocab_size, self.d_embed)
        self.target_embedding = torch.nn.Embedding(fr_vocab_size, self.d_embed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.position_encoding = PositionalEncoding(self.d_model).to(self.device)
        self.encoder_block = torch.nn.ModuleList([Encoder(self.d_embed, self.d_embed, self.n_heads, self.d_model) for _ in range(n_layers)])
        self.encoder_block = torch.nn.Sequential(*self.encoder_block)
        self.encoder_block.to(self.device)
        self.decoder_block = [Decoder(self.d_embed, self.d_embed, self.n_heads, self.d_model) for _ in range(n_layers)]
        [decoder.to(self.device) for decoder in self.decoder_block]
        self.linear = torch.nn.Linear(self.d_model, self.vocab_size)
        self.softmax = torch.nn.Softmax()

    def forward(self, inputs, targets):
        e_inputs = self.input_embedding(inputs)
        e_inputs = self.position_encoding(e_inputs)
        d_inputs = self.target_embedding(targets)  # Shift right NEEDED
        d_inputs = self.position_encoding(d_inputs)
        encoder_output = self.encoder_block(e_inputs)
        # self.encoder_block(e_inputs)
        decoder_output = d_inputs
        for i, decoder in enumerate(self.decoder_block):
            decoder_output = decoder(encoder_output, decoder_output)
        return self.linear(decoder_output)


class NMT:
    def __init__(self):
        """-----------------------------------------------------------------------
        Data config
        -----------------------------------------------------------------------"""
        PATH = r"C:\Users\Rajkumar\Downloads"
        DATA_DIR = PATH + r"\NMT_Data"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        RANDOM_STATE = 123
        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        """-----------------------------------------------------------------------
        Model config
        -----------------------------------------------------------------------"""
        self.MAX_INPUT_LEN = 14
        self.MAX_OUTPUT_LEN = 14
        self.n_heads = 6
        self.n_layers = 2
        self.d_model = 512
        self.LR = 1e-5
        self.N_EPOCHS = 10
        self.BATCH_SIZE = 10

        def clean_text(sentence):
            text = sentence.lower()  # normalize
            text = re.sub(r"[.!,\ ]*", "", text)  # remove special chars
            english, french = text.split("\t")  # split english from french.
            french = re.findall(r"(\b[^\s]+\b)", french)[0]
            return english, french


        data = None
        with open("data/eng-fra.txt", "rb") as f:
            data = f.read()
            data = data.decode("utf8")
        data = data.split("\n")
        english = []
        french = []
        english_tmp_vocab = set()
        french_tmp_vocab = set()
        french_reverse = {}
        en_max_input_len = 0
        fr_max_input_len = 0
        for idx, sentence in enumerate(data):
            if sentence == '':
                continue
            tmp_eng, tmp_fr = clean_text(sentence)
            english.append(tmp_eng)
            tmp_tokens = set(word_tokenize(tmp_eng))
            if len(tmp_tokens) > en_max_input_len:
                en_max_input_len = len(tmp_tokens)
            english_tmp_vocab = english_tmp_vocab | tmp_tokens
            french.append(tmp_fr)
            tmp_tokens = set(word_tokenize(tmp_fr))
            french_tmp_vocab = french_tmp_vocab | tmp_tokens
            if len(tmp_tokens) > fr_max_input_len:
                fr_max_input_len = len(tmp_tokens)

        offset = 2
        english_vocab = {key:int(value) for value, key in enumerate(english_tmp_vocab)}
        self.vocab_size = len(english_vocab)

        french_vocab = {key:int(value+offset) for value, key in enumerate(french_tmp_vocab)}
        french_vocab["<SOS>"] = 0
        french_vocab["<EOS>"] = 1
        self.fr_vocab_size = len(french_vocab)
        french_reverse = {key:value for value, key in french_vocab.items()}

        df = pd.DataFrame({"english": english, "french": french})
        tokenizer = get_tokenizer("basic_english", language="en")

        tr_df, ts_df = train_test_split(df, test_size=0.3)
        tr_dataset = NMTDataset(tr_df, tokenizer, english_vocab, french_vocab,
                                self.MAX_INPUT_LEN, self.MAX_OUTPUT_LEN, len(english_vocab),
                                len(french_vocab))
        ts_dataset = NMTDataset(ts_df, tokenizer, english_vocab, french_vocab,
                                self.MAX_INPUT_LEN, self.MAX_OUTPUT_LEN, len(english_vocab),
                                len(french_vocab))

        def collate_tensor_fn(batch):
            en_tokens = torch.stack([batch[i][0] for i in range(self.BATCH_SIZE)], dim=0)
            fr_tokens = torch.stack([batch[i][1] for i in range(self.BATCH_SIZE)], dim=0)
            return en_tokens.squeeze(), fr_tokens.squeeze()

        self.tr_dloader = DataLoader(tr_dataset, batch_size=self.BATCH_SIZE, shuffle=True,
                                     collate_fn=collate_tensor_fn
                                     #num_workers=self.N_WORKERS, batch_sampler=BatchSampler(SequentialSampler(tr_dataset)),
                                     )

        self.vl_dloader = DataLoader(ts_dataset, batch_size=self.BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_tensor_fn
                                     #num_workers=self.N_WORKERS, batch_sampler=BatchSampler(SequentialSampler(tr_dataset)),
                                     )
        for x, y in self.tr_dloader:
            break
        self.tr_size = len(tr_df)

    def build_model(self):
        model = Transformer(self.n_heads, self.n_layers, self.d_model, self.vocab_size, self.fr_vocab_size)
        model.to(self.device)
        return model

    def fit(self, model):
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        steps_train = self.tr_size//self.BATCH_SIZE

        for epoch in range(self.N_EPOCHS):
            model.train()
            loss_train = 0

            with tqdm(total=self.tr_size, desc=f"Epoch {epoch}") as pbar:
                for en_tokens, fr_tokens in self.tr_dloader:
                    en_tokens = en_tokens.to(self.device)
                    fr_tokens = fr_tokens.to(self.device)
                    fr_prediction = model(en_tokens, fr_tokens)
                    loss = criterion(torch.tensor(torch.argmax(fr_prediction, dim=-1), requires_grad=True, dtype=torch.float32),
                                     fr_tokens.to(torch.float32))
                    loss.backward()
                    loss_train += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(self.BATCH_SIZE)
                    pbar.set_postfix_str("Train Loss: {:.5f}".format(loss_train/steps_train))


if __name__ == "__main__":
    nmt = NMT()
    model = nmt.build_model()
    nmt.fit(model)
    print('debug checkpoint...')


















