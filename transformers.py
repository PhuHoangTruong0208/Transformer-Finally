import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# tính toán điểm attention và giá trị attention thông qua 3 ma trận được tách ra từ đầu ra của lớp Linear
# input: 3 x d_model -> Q, K, V
def compute_qkv(q, k, v, mask):
    d_k = q.size()[-1]
    scaled = torch.matmul(input=q, other=k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask != None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(input=scaled, dim=-1)
    values = torch.matmul(input=attention, other=v)
    return values, attention

# mã hóa vị trí giúp các vector tokens chứa các mã hóa vị trí giúp mô hình nhớ được vị trí của từng từ
def position_encoding(d_model, max_sequence_length):
    even_i = torch.arange(start=0, end=d_model, step=2, dtype=torch.float32)
    denominator = torch.pow(10000, even_i/d_model)
    position = torch.arange(max_sequence_length).reshape(max_sequence_length, 1)
    even_PE = torch.sin(position / denominator)
    odd_PE = torch.cos(position / denominator)
    stacked = torch.stack([even_PE, odd_PE], dim=2)
    return stacked.reshape(max_sequence_length, d_model)

# tính toán giá trị chú ý của đầu vào (dùng trong cả encoder và decoder)
class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dims = d_model // num_heads
        self.qkv_linear = nn.Linear(in_features=d_model, out_features=3 * d_model).to(device)
        self.out_linear = nn.Linear(in_features=d_model, out_features=d_model).to(device)

    def forward(self, x, mask):
        batch_size, max_sequence_length, input_dims = x.size()
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dims)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v, = qkv.chunk(3, dim=-1)
        values, attention = compute_qkv(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, max_sequence_length, self.d_model)
        out = self.out_linear(values)
        return out
    

# tính toán giá trị chú ý của (một phần của decoder) nó sẽ dùng output của encoder để tính toán
# kết quả y dự đoán
class MultiheadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dims = d_model // num_heads
        self.q_linear = nn.Linear(in_features=d_model, out_features=d_model).to(device)
        self.kv_linear = nn.Linear(in_features=d_model, out_features=2 * d_model).to(device)
        self.out_linear = nn.Linear(in_features=d_model, out_features=d_model).to(device)

    def forward(self, x, y, mask):
        batch_size, max_sequence_length, d_model = x.size()
        q = self.q_linear(x)
        kv = self.kv_linear(y)
        q = q.reshape(batch_size, max_sequence_length, self.num_heads, self.head_dims)
        kv = kv.reshape(batch_size, max_sequence_length, self.num_heads, 2 * self.head_dims)
        q = q.permute(0, 2, 1, 3)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = compute_qkv(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, max_sequence_length, d_model)
        out = self.out_linear(values)
        return out

# batch_size = 30
# max_sequence_length = 200
# d_model = 512
# num_heads = 8

# x = torch.rand(batch_size, max_sequence_length, d_model)
# attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
# print(attention(x, mask=None))
    

class LayerNorm(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, dropout):
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=ffn_hidden).to(device)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(in_features=ffn_hidden, out_features=d_model).to(device)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x
    
# chuyển lô thành các tokens để chuẩn bị đưa vào chạy qua mô hình
class SentenceEmbedding(nn.Module):

    def __init__(self, d_model, max_sequence_lenght, to_index, dropout, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_lenght = max_sequence_lenght
        self.char_to_index = to_index
        self.dropout = dropout
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.embedding = nn.Embedding(num_embeddings=len(to_index), embedding_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            tokens_sentence = [self.char_to_index[token] for token in list(sentence)]
            if start_token:
                tokens_sentence.insert(0, self.char_to_index[self.START_TOKEN])
            if end_token:
                tokens_sentence.append(self.char_to_index[self.END_TOKEN])
            for _ in range(len(tokens_sentence), self.max_sequence_lenght):
                tokens_sentence.append(self.char_to_index[self.PADDING_TOKEN])
            if len(tokens_sentence) > self.max_sequence_lenght:
                tokens_sentence = tokens_sentence[:self.max_sequence_lenght]
            return torch.tensor(tokens_sentence)
        
        tokenized = []
        for sentence in batch:
            tokenized.append(tokenize(sentence, start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(device)
    
    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        x = self.dropout(x + position_encoding(self.d_model, self.max_sequence_lenght).to(device))
        return x

# x = ("hello", "what is your name?")
# index_to = dict(enumerate(["h", "e", "l", "o", "w", "a", "t", "y", "o", "u", "r", "n", "m", "e", "?", " ", "i", "s"]+["<s>", "<e>", "<p>"]))
# to_index = {v:k for k, v in index_to.items()}
# embed = SentenceEmbedding(d_model=512, max_sequence_lenght=200, char_to_index=to_index, dropout=0.1,
#                           START_TOKEN="<s>", END_TOKEN="<e>", PADDING_TOKEN="<p>")
# print(embed(x, start_token=True, end_token=True))
    
class CreateMask(nn.Module):
    def __init__(self, max_sequence_length, inf=-1e9):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.inf = inf

    def forward(self, source_batch, target_batch):
        num_sentences = len(source_batch)
        look_ahead_mask = torch.full([self.max_sequence_length, self.max_sequence_length] , True)
        look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
        encoder_padding_mask = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)
        decoder_padding_mask_self_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)
        decoder_padding_mask_cross_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length] , False)

        for idx in range(num_sentences):
            eng_sentence_length, kn_sentence_length = len(source_batch[idx]), len(target_batch[idx])
            eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, self.max_sequence_length)
            kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, self.max_sequence_length)
            encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
            encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
            decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
            decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
            decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
            decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

        encoder_self_attention_mask = torch.where(encoder_padding_mask, self.inf, 0)
        decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, self.inf, 0)
        decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, self.inf, 0)
        return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

# create_mask = CreateMask(max_sequence_length=200)
# encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_mask(("hello", "what is your name?"), ("hi", "my name is nimy"))
# print(encoder_self_attention_mask)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout):
        super().__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = LayerNorm(parameters_shape=[d_model]).to(device)
        self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layernorm2 = LayerNorm(parameters_shape=[d_model]).to(device)
    
    def forward(self, x, encoder_mask):
        pre_x = x.clone()
        x = self.attention(x, mask=encoder_mask)
        x = self.dropout1(x)
        x = self.layernorm1(x + pre_x)
        pre_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.layernorm2(x + pre_x)
        return x


class SequentialEnocder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout, num_layers):
        super().__init__()
        self.layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout)
                  for _ in range(num_layers)]
    
    def forward(self, x, encoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_mask=encoder_mask)
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout, num_layers, to_index, max_sequence_length, START_TOKEN, 
                END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.embedding = SentenceEmbedding(d_model=d_model, max_sequence_lenght=max_sequence_length, to_index=to_index,
                                    dropout=dropout, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, PADDING_TOKEN=PADDING_TOKEN)
        self.encoder_layer = SequentialEnocder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout, num_layers=num_layers)

    def forward(self, x, encoder_mask, start_token, end_token):
        x = self.embedding(x, start_token=start_token, end_token=end_token)
        x = self.encoder_layer(x, encoder_mask=encoder_mask)
        return x


# x = ("hello", "what is your name?")
# index_to = dict(enumerate(["h", "e", "l", "o", "w", "a", "t", "y", "o", "u", "r", "n", "m", "e", "?", " ", "i", "s"]+["<s>", "<e>", "<p>"]))
# to_index = {v:k for k, v in index_to.items()}

# enc = Encoder(d_model=512, num_heads=8, ffn_hidden=1024, dropout=0.1, num_layers=2, to_index=to_index,
#               max_sequence_length=100, START_TOKEN="<s>", END_TOKEN="<e>", PADDING_TOKEN="<p>")
# result = enc(x, encoder_mask=None, start_token=False, end_token=True)
# print(result)
    

class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden, dropout):
        super().__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout)
        self.layernorm1 = LayerNorm(parameters_shape=[d_model]).to(device)
        self.cross_attention = MultiheadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layernorm2 = LayerNorm(parameters_shape=[d_model]).to(device)
        self.ffn = PositionwiseFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, dropout=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.layernorm3 = LayerNorm(parameters_shape=[d_model]).to(device)
    
    def forward(self, x, y, decoder_mask, decoder_cross_mask):
        pre_y = y.clone()
        y = self.attention(y, mask=decoder_mask)
        y = self.dropout1(y)
        y = self.layernorm1(y + pre_y)
        pre_y = y.clone()
        y = self.cross_attention(x, y, mask=decoder_cross_mask)
        y = self.dropout2(y)
        y = self.layernorm2(y + pre_y)
        pre_y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layernorm3(y + pre_y)
        return y


class SequentialDecoder(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden, dropout, num_layers):
        super().__init__()
        self.layers = [DecoderLayer(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout)
                       for _ in range(num_layers)]
        
    def forward(self, x, y, decoder_mask, decoder_cross_mask):
        for layer in self.layers:
            y = layer(x, y, decoder_mask=decoder_mask, decoder_cross_mask=decoder_cross_mask)
        return y
    

class Decoder(nn.Module):
    
    def __init__(self, d_model, num_heads, ffn_hidden, dropout, num_layers, to_index, max_sequence_length,
                START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.embedding = SentenceEmbedding(d_model=d_model, max_sequence_lenght=max_sequence_length, to_index=to_index,
                                    dropout=dropout, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, PADDING_TOKEN=PADDING_TOKEN)
        self.decoder_layer = SequentialDecoder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden,
                                        dropout=dropout, num_layers=num_layers)
    
    def forward(self, x, y, decoder_mask, decoder_cross_mask, start_token, end_token):
        y = self.embedding(y, start_token=start_token, end_token=end_token)
        y = self.decoder_layer(x, y, decoder_mask=decoder_mask, decoder_cross_mask=decoder_cross_mask)
        return y
    

# y = ("hello", "what is your name?")
# x = torch.randn(2, 100, 512)
# index_to = dict(enumerate(["h", "e", "l", "o", "w", "a", "t", "y", "o", "u", "r", "n", "m", "e", "?", " ", "i", "s"]+["<s>", "<e>", "<p>"]))
# to_index = {v:k for k, v in index_to.items()}

# dec = Decoder(d_model=512, num_heads=8, ffn_hidden=1024, dropout=0.1, num_layers=2, to_index=to_index,
#               max_sequence_length=100, START_TOKEN="<s>", END_TOKEN="<e>", PADDING_TOKEN="<p>")
# result = dec(x, y, decoder_mask=None, decoder_cross_mask=None, start_token=False, end_token=True)
# print(result)
    

class TranformersModel(nn.Module):

    def __init__(self, d_model, num_heads, ffn_hidden, dropout, to_index, max_sequence_length, num_layers, START_TOKEN, vocab_size,
                END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.encoder_layer = Encoder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout,
                            num_layers=num_layers, to_index=to_index, max_sequence_length=max_sequence_length, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, PADDING_TOKEN=PADDING_TOKEN)
        self.decoder_layer = Decoder(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout,
                            num_layers=num_layers, to_index=to_index, max_sequence_length=max_sequence_length, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, PADDING_TOKEN=PADDING_TOKEN)
        self.out_linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x, y, encoder_mask, decoder_mask, decoder_cross_mask, enc_start_token, enc_end_token,
                dec_start_token, dec_end_token):
        x = self.encoder_layer(x, encoder_mask=encoder_mask, start_token=enc_start_token, end_token=enc_end_token)
        y = self.decoder_layer(x, y, decoder_mask=decoder_mask, decoder_cross_mask=decoder_cross_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.out_linear(y)
        return out
    
# x = ("hello", "what is your name?")
# y = ("hi", "my name is nimy")
# index_to = dict(enumerate(["h", "e", "l", "o", "w", "a", "t", "y", "o", "u", "r", "n", "m", "e", "?", " ", "i", "s"]+["<s>", "<e>", "<p>"]))
# to_index = {v:k for k, v in index_to.items()}

# dec = TranformersModel(d_model=512, num_heads=8, ffn_hidden=1024, dropout=0.1, num_layers=2, to_index=to_index,
#               max_sequence_length=100, START_TOKEN="<s>", END_TOKEN="<e>", PADDING_TOKEN="<p>")
# result = dec(x, y, encoder_mask=None, decoder_mask=None, decoder_cross_mask=None, enc_start_token=False, enc_end_token=True,
#              dec_start_token=False, dec_end_token=False)
# print(result)