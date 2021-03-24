import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

def create_padding_mask(seq, padding_idx):
    return (seq == padding_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((1, len_s, len_s)), diagonal=1).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        angles = self.get_angles(torch.arange(0., max_len).unsqueeze(1),
                                torch.arange(0.,d_model).unsqueeze(0),
                                d_model)
        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])
        self.pos_encoding = angles.unsqueeze(0)

    def get_angles(self, pos, i, d_model):
        angles = 1 / torch.pow(10000., (2*(i//2)/d_model))
        return pos*angles


    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dense_q = nn.Linear(d_model, d_model)
        self.dense_k = nn.Linear(d_model, d_model)
        self.dense_v = nn.Linear(d_model, d_model)
        
        assert d_model % n_heads == 0

        self.output_linear = nn.Linear(d_model, d_model)
    
    def split_heads(self, inputs, batch_size):
        shape = (batch_size, -1, self.n_heads, self.d_model // self.n_heads)
        split_inputs = inputs.reshape(shape)
        return split_inputs.permute(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, query, key, value, mask):
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / np.sqrt(self.d_model // self.n_heads)
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool)
            scores.masked_fill_(mask, -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value)
        return context, scores
    
    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        query = self.dense_q(query)
        key = self.dense_k(key)
        value = self.dense_v(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        context, attention = self.scaled_dot_product_attention(query, key, value, mask)

        context = context.permute(0, 2, 1, 3)
        context = context.reshape(batch_size, -1, self.d_model)

        output = self.output_linear(context)

        return output, attention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_units, dropout):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        self.dense_1 = nn.Linear(d_model, ffn_units)
        self.dense_2 = nn.Linear(ffn_units, d_model)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, inputs, mask):
        attention, _ = self.multi_head_attention(inputs, inputs, inputs, mask)
        attention = self.dropout_1(attention)
        attention = self.norm_1(attention+inputs)

        outputs = F.gelu(self.dense_1(attention))
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs)
        outputs = self.norm_2(outputs+attention)
        return outputs

        
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_units, dropout):
        super().__init__()
        self.multi_head_attention_1 = MultiHeadAttention(d_model, n_heads)
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)

        self.multi_head_attention_2 = MultiHeadAttention(d_model, n_heads)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dense_1 = nn.Linear(d_model, ffn_units)
        self.dense_2 = nn.Linear(ffn_units, d_model)
        self.dropout_3 = nn.Dropout(dropout)
        self.norm_3 = nn.LayerNorm(d_model)

    
    def forward(self, inputs, enc_outputs, lookahead_mask, padding_mask):
        attention_1, _ = self.multi_head_attention_1(inputs, inputs, inputs, lookahead_mask)
        attention_1 = self.dropout_1(attention_1)
        attention_1 = self.norm_1(attention_1+inputs)

        attention_2, _ = self.multi_head_attention_2(attention_1, enc_outputs, enc_outputs, padding_mask)
        attention_2 = self.dropout_2(attention_2)
        attention_2 = self.norm_2(attention_2+attention_1)

        outputs = F.gelu(self.dense_1(attention_2))
        outputs = self.dense_2(outputs)
        outputs = self.dropout_3(outputs)
        outputs = self.norm_3(outputs)
        return outputs


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, ffn_units, dropout):
        super().__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ffn_units, dropout) for i in range(n_layers)])

    def forward(self, x, mask):
        enc_output = x
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, mask)
        return enc_output

class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, ffn_units, dropout):
        super().__init__()
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ffn_units, dropout) for i in range(n_layers)])

    def forward(self, x, enc_outputs, lookahead_mask, padding_mask):
        dec_output = x
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_outputs, lookahead_mask, padding_mask)
        return dec_output


class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, ffn_units, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, n_layers, n_heads, ffn_units, dropout)
        self.decoder = Decoder(d_model, n_layers, n_heads, ffn_units, dropout)


    def forward(self, enc_inputs, dec_inputs, enc_mask, dec_lookahead_mask, dec_padding_mask):
        enc_outputs = self.encoder(enc_inputs, enc_mask)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_lookahead_mask, dec_padding_mask)
        return dec_outputs

X = torch.rand(1, 50, 128)
X2 = torch.rand(1, 50, 128)
transformer = Transformer(128, 2,  8, 1024, 0.1)
y = transformer(X, X2, None, None, None)

y.shape