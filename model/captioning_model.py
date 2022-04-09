import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder

class CaptioningModel(nn.Module):
    def __init__(self, embed_dim:int=256, encoder_type:str='resnet152', encoder_pretrained:bool=True,
                decoder_type:str='gru', hidden_dim:int=512, nhead:int=8, num_layers:int=1, bidirectional:bool=True,
                vocab_size:int=8000, max_seq_len:int=8000, dropout:float=0.3):
        super(CaptioningModel, self).__init__()

        self.encoder = Encoder(output_dim=embed_dim, encoder_type=encoder_type, 
                               encoder_pretrained=encoder_pretrained)
        self.decoder = Decoder(embed_dim=embed_dim, decoder_type=decoder_type, 
                               hidden_dim=hidden_dim, nhead=nhead, 
                               num_layers=num_layers, bidirectional=bidirectional, 
                               vocab_size=vocab_size, max_seq_len=max_seq_len)
    
    def forward(self, images:torch.Tensor, caption_ids:torch.Tensor, lengths:int):
        """
        Args:
            images (torch.Tensor): [batch_size, 3, 224, 224]
            caption_ids (torch.Tensor): [batch_size, max_seq_len]
            lengths (torch.Tensor): [batch_size]
        """

        features = self.encoder(images) # (batch_size, output_dim)
        logits = self.decoder(features, caption_ids, lengths) # (batch_size, max_seq_length, vocab_size)
        log_prob = F.log_softmax(logits, dim=-1) # (batch_size, max_seq_length, vocab_size)
        
        return log_prob