import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SingleLogEncoder(nn.Module):
    def __init__(self, embedding_dim=768, transformer_layers=2, n_heads=12, dropout=0.1):
        super(SingleLogEncoder, self).__init__()
        
        # Bert Tokenizer and Model for word embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Transformer Encoder Layer
        self.transformer_encoder = nn.ModuleList([
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
        
        # Dimension of the BERT embeddings
        self.embedding_dim = embedding_dim

    def forward(self, tokens):
        # Tokenize the log message
        # tokens = self.tokenizer(log_message, return_tensors='pt', truncation=True, padding=True)
        
        # Get BERT embeddings
        with torch.no_grad():
            bert_output = self.bert(**tokens)
        
        # Take the last hidden state of the BERT model
        embeddings = bert_output.last_hidden_state
        
        # Transformer Encoder expects input of shape (sequence_length, batch_size, embedding_dim)
        embeddings = embeddings.permute(1, 0, 2)  # Permute to (sequence_length, batch_size, embedding_dim)
        
        # Pass through transformer encoder layers
        for layer in self.transformer_encoder:
            embeddings = layer(embeddings)
        
        
        # Apply Feed Forward Neural Network (FFN)
        ffn_output = self.ffn(embeddings)
        
        # Pooling to get fixed-size vector
        pooled_output = self.pooling(ffn_output.permute(1, 2, 0)).squeeze(-1)
        
        return pooled_output

    def tokenize(self, log_message):
        return self.tokenizer(log_message, return_tensors='pt', truncation=True, padding=True)


def test_single_log_encoder():
    embedding_dim = 768  # BERT base model dimension
    transformer_layers = 2
    n_heads = 12
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_encoder = SingleLogEncoder(embedding_dim, transformer_layers, n_heads, dropout).to(device)
    log_message = "Received block blk_-2856928563366064757 of size 67108864 from /10.251.42.9"
    tokens = log_encoder.tokenize(log_message)
    tokens = tokens.to(device)

    log_representation = log_encoder(tokens)

    print(log_representation.shape)  # Should output (batch_size, embedding_dim)

if __name__ == '__main__':
    test_single_log_encoder()