import torch
import torch.nn as nn
from models import SingleLogEncoder
from preprocessing import parse_dataset, load_dataset

class LogSequenceEncoder(nn.Module):
    def __init__(self, embedding_dim=768, transformer_layers=2, n_heads=12, dropout=0.1):
        super(LogSequenceEncoder, self).__init__()
        
        # Single Log Encoder
        self.single_log_encoder = SingleLogEncoder(embedding_dim, transformer_layers, n_heads, dropout)
        
        # Transformer Encoder Layer for sequence level
        self.sequence_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dropout=dropout)
            for _ in range(transformer_layers)
        ])

        # Feed Forward Neural Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, log_messages):
        # Encode each log message using Single Log Encoder
        log_representations = [self.single_log_encoder(log) for log in log_messages]
        
        # Stack log representations to form a sequence
        log_representations = torch.stack(log_representations).permute(1, 0, 2)  # Shape: (sequence_length, batch_size, embedding_dim)
        
        # Pass through transformer encoder layers
        for layer in self.sequence_transformer_encoder:
            log_representations = layer(log_representations)
        
        # Apply Feed Forward Neural Network (FFN)
        ffn_output = self.ffn(log_representations)

        # Pooling to get fixed-size vector
        pooled_output = self.pooling(ffn_output.permute(1, 2, 0)).squeeze(-1)
        
        return pooled_output


if __name__ == "__main__":
    embedding_dim = 768  # BERT base model dimension
    transformer_layers = 2
    n_heads = 12
    dropout = 0.1
    
    log_sequence_encoder = LogSequenceEncoder(embedding_dim, transformer_layers, n_heads, dropout)
    dataset = load_dataset("loghub/Linux/Linux_2k.log")
    results, template_miner = parse_dataset(dataset)
    log_messages = [result["template_mined"] for result in results]
    log_sequence_representation = log_sequence_encoder(log_messages)
    
    print(log_sequence_representation.shape)  # Should output (batch_size, embedding_dim)
