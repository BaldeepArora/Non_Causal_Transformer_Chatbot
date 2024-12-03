

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from transformers import BertTokenizer

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NonCausalTransformer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, ff_size, num_classes, max_seq_len, dropout=0.1):
        super(NonCausalTransformer, self).__init__()
        
        # Token embeddings (BERT embeddings not used; random embeddings learned from scratch)
        self.token_embeddings = nn.Embedding(tokenizer.vocab_size, hidden_size)
        
        # Relative positional encodings
        self.relative_pos_enc = nn.Embedding(2 * max_seq_len - 1, hidden_size)
        self.max_seq_len = max_seq_len
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def add_relative_position_encodings(self, embeddings):
        batch_size, seq_len, hidden_size = embeddings.size()

        # Create relative position matrix
        positions = torch.arange(seq_len, dtype=torch.long, device=embeddings.device).unsqueeze(0)
        rel_pos_matrix = positions - positions.T + self.max_seq_len - 1

        # Clamp indices to the valid range of [0, 2 * max_seq_len - 2]
        rel_pos_matrix = rel_pos_matrix.clamp(0, 2 * self.max_seq_len - 2)

        # Get relative positional encodings
        rel_pos_encodings = self.relative_pos_enc(rel_pos_matrix)  # Shape: (seq_len, seq_len, hidden_size)

        # Expand dimensions to match embeddings
        rel_pos_encodings = rel_pos_encodings.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Shape: (batch_size, seq_len, seq_len, hidden_size)

        # Aggregate relative positional encodings across the sequence
        rel_pos_encodings = rel_pos_encodings.mean(dim=2)  # Shape: (batch_size, seq_len, hidden_size)

        # Add to embeddings
        return embeddings + rel_pos_encodings


    def forward(self, input_ids, attention_mask, decoder_input_ids):
        # Token embeddings
        embeddings = self.token_embeddings(input_ids)
        embeddings = self.add_relative_position_encodings(embeddings)
        
        # Decoder input embeddings
        decoder_embeddings = self.token_embeddings(decoder_input_ids)
        
        # Transformer
        transformer_output = self.transformer(
            src=embeddings,
            tgt=decoder_embeddings,
            src_key_padding_mask=~attention_mask.bool()
        )
        
        # Fully connected output layer
        output = self.fc_out(transformer_output)
        return output
