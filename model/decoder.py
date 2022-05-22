import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Decoder(nn.Module):
    def __init__(self, embed_dim:int, decoder_type:str,
                 hidden_dim:int, nhead:int, num_layers:int, bidirectional:bool,
                 vocab_size:int, max_seq_len:int=20):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.decoder_type = decoder_type.lower()
        self.max_seq_len = max_seq_len

        # Define the decoder
        if self.decoder_type == 'rnn':
            self.decoder = nn.RNN(embed_dim * 2, hidden_dim, num_layers,
                                  bidirectional=bidirectional, batch_first=False)
        elif self.decoder_type == 'lstm':
            self.decoder = nn.LSTM(embed_dim * 2, hidden_dim, num_layers,
                                   bidirectional=bidirectional, batch_first=False)
        elif self.decoder_type == 'gru':
            self.decoder = nn.GRU(embed_dim * 2, hidden_dim, num_layers,
                                  bidirectional=bidirectional, batch_first=False)
        elif self.decoder_type == 'transformer':
            decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
            self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
            self.pos_embed1D = torch.nn.Parameter(torch.randn(max_seq_len, embed_dim))
        else:
            raise NotImplementedError(f'Decoder type {decoder_type} is not implemented')
        
        # Define the output layer
        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.rnn_output_to_embed_linear = nn.Linear(rnn_out_dim, embed_dim)

        self.decoder_linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.activation = nn.GELU()
        self.decoder_linear2 = nn.Linear(embed_dim * 4, vocab_size)
        self.out = nn.Sequential(self.decoder_linear1, self.activation, self.decoder_linear2)

    def forward(self, features, caption_ids):
        decoder_embedding = self.embed(caption_ids) # (batch_size, max_seq_len-1, embed_dim)
        features = features.unsqueeze(1) # (batch_size, 1, embed_dim)
        features = features.repeat(1, decoder_embedding.size(1), 1) # (batch_size, max_seq_len-1, embed_dim)

        if self.decoder_type == 'transformer':
            #decoder_input = decoder_embedding + features # (batch_size, max_seq_len, embed_dim)
            #decoder_input = decoder_input + self.pos_encoder(decoder_input) # (batch_size, max_seq_len, embed_dim)
            decoder_input = decoder_embedding + features + self.pos_embed1D[:self.max_seq_len-1, :] # (batch_size, max_seq_len-1, embed_dim)
            decoder_input = decoder_input.permute(1, 0, 2) # (max_seq_len-1, batch_size, embed_dim)
            features = features.permute(1, 0, 2) # (max_seq_len-1, batch_size, embed_dim)

            trg_mask = self.generate_square_subsequent_mask(decoder_input.size(0), device=decoder_input.device) # (max_seq_len-1, max_seq_len-1)
            trg_key_padding_mask = (caption_ids == 0) # (batch_size, max_seq_len-1)

            decoder_output = self.decoder(decoder_input, features,
                                          tgt_mask=trg_mask, tgt_key_padding_mask=trg_key_padding_mask) # (max_seq_len-1, batch_size, embed_dim)      
        else: # rnn, lstm, gru
            # Addition or concatenation?
            # decoder_input = decoder_embedding + features # (batch_size, max_seq_len-1, embed_dim)
            decoder_input = torch.cat((features, decoder_embedding), dim=2) # (batch_size, max_seq_len-1, embed_dim * 2)
            decoder_input = decoder_input.permute(1, 0, 2) # (max_seq_len-1, batch_size, embed_dim * 2)

            decoder_input = pack_padded_sequence(decoder_input, self.get_non_pad_length(caption_ids), batch_first=False)
            decoder_output, _ = self.decoder(decoder_input) # decoder_output: (max_seq_len-1, batch_size, bidirectional * hidden_dim)
            decoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(decoder_output, total_length=self.max_seq_len-1, batch_first=False)
        
            decoder_output = self.rnn_output_to_embed_linear(decoder_output) # (max_seq_len-1, batch_size, embed_dim)
        
        decoder_output = decoder_output.permute(1, 0, 2) # (batch_size, max_seq_len-1, embed_dim)
        decoder_logits = self.out(decoder_output) # (batch_size, max_seq_len-1, vocab_size)

        return decoder_logits

    def get_non_pad_length(self, caption_ids:torch.Tensor):
        return caption_ids.ne(0).sum(dim=1).to('cpu')

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe.requires_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]