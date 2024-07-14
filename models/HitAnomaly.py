import torch
import torch.nn as nn
from models import SingleLogEncoder, LogSequenceEncoder, ParameterEncoder

class LogAnomalyDetector(nn.Module):
    def __init__(self, embedding_dim=768, transformer_layers=2, n_heads=12, dropout=0.1):
        super(LogAnomalyDetector, self).__init__()
        self.log_sequence_encoder = LogSequenceEncoder(embedding_dim, transformer_layers, n_heads, dropout)
        self.parameter_encoder = ParameterEncoder(embedding_dim, transformer_layers, n_heads, dropout)
        
        # Attention Layer Parameters
        self.W_alpha = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.v = nn.Parameter(torch.Tensor(embedding_dim, 1))
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    
    def attention_score(self, x):
        return torch.exp(torch.matmul(torch.tanh(torch.matmul(x, self.W_alpha)), self.v))

    def forward(self, log_messages, log_template_groups):
        R_s = self.log_sequence_encoder(log_messages)
        R_p = self.parameter_encoder(log_template_groups)
        
        # Compute attention scores
        f_R_s = self.attention_score(R_s)
        f_R_p = self.attention_score(R_p)
        
        alpha_s = f_R_s / (f_R_s + f_R_p)
        alpha_p = f_R_p / (f_R_s + f_R_p)
        
        # Compute weighted sum
        combined_representation = alpha_s * R_s + alpha_p * R_p
        
        # Classifier
        output = self.classifier(combined_representation)
        
        return output
