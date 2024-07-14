import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from models import SingleLogEncoder
from preprocessing import parse_dataset, load_dataset

# Similar to SingleLogEncoder, we will use BERT for word embeddings
class SingleParaEncoder(nn.Module):
    def __init__(self, embedding_dim=768, transformer_layers=2, n_heads=12, dropout=0.1):
        super(SingleParaEncoder, self).__init__()
        
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

    def forward(self, para_tokens):
        with torch.no_grad():
            bert_output = self.bert(**para_tokens)

        embeddings = bert_output.last_hidden_state

        embeddings = embeddings.permute(1, 0, 2)

        for layer in self.transformer_encoder:
            embeddings = layer(embeddings)

        ffn_output = self.ffn(embeddings)

        pooled_output = self.pooling(ffn_output.permute(1, 2, 0)).squeeze(-1)

        return pooled_output

    def tokenize(self, parameter):
        return self.tokenizer(parameter, return_tensors='pt', truncation=True, padding=True)

class ParameterEncoder(nn.Module):
    def __init__(self, embedding_dim=768, transformer_layers=2, n_heads=12, dropout=0.1, device='cpu'):
        super(ParameterEncoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Single Parameter Encoder
        self.single_para_encoder = SingleParaEncoder(embedding_dim, transformer_layers, n_heads, dropout)
        self.para_pooling = nn.AdaptiveAvgPool1d(1)

        # Log Sequence Encoder
        self.log_encoder = SingleLogEncoder(embedding_dim, transformer_layers, n_heads, dropout)
        self.log_pooling = nn.AdaptiveAvgPool1d(1)

        # Transformer Encoder Layer for sequence level
        self.sequence_transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dropout=dropout)
            for _ in range(transformer_layers)
        ])
        
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.final_pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, log_template_tokens, para_tokens):
        combined_output = []

        for log_template, parameters in zip(log_template_tokens, para_tokens):
            log_representation = self.log_encoder(log_template)
            
            para_representations = [self.single_para_encoder(para) for para in parameters]
            if para_representations:
                para_representations = torch.stack(para_representations).permute(1, 0, 2).squeeze(0)
                para_pooled = torch.mean(para_representations, dim=0, keepdim=True)
            else:
                # Handle the case where para_representations is empty.
                # Option 1: Skip this group of parameters.
                # Option 2: Create a default tensor. Example:
                para_pooled = torch.zeros([self.embedding_dim], dtype=torch.float32).to(self.device)
                # Adjust `expected_dimension` as per your model's requirements.

            combined_output.append(self.linear(para_pooled) + self.linear(log_representation))
        
        combined_output = torch.stack(combined_output).permute(1, 0, 2)
        
        for layer in self.sequence_transformer_encoder:
            combined_output = layer(combined_output)
        
        final_output = self.final_pooling(combined_output.permute(1, 2, 0)).squeeze(-1)

        return final_output
    
    def tokenize(self, log_template_groups, device):
        log_template_tokens = [self.log_encoder.tokenize(log_template["log_template"]) for log_template in log_template_groups]
        para_tokens = [[self.single_para_encoder.tokenize(para) for para in log_template_group["parameters"]] for log_template_group in log_template_groups]
        return [token.to(device) for token in log_template_tokens], [[token.to(device) for token in para_token] for para_token in para_tokens]

def group_parameters(logs):
    log_template_groups = {}
    for log in logs:
        log_template = log["template_mined"]
        parameters = log["parameters"]

        if log_template not in log_template_groups:
            log_template_groups[log_template] = []
        
        if parameters and len(parameters) > 0:
          log_template_groups[log_template].append(
              [parameter.value for parameter in parameters]
          )
        
    return [{"log_template": log_template, "parameters": parameters} for log_template, parameters in log_template_groups.items()]


def test_parameter_encoder():
    embedding_dim = 768  # BERT base model dimension
    transformer_layers = 2
    n_heads = 12
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parameter_encoder = ParameterEncoder(embedding_dim, transformer_layers, n_heads, dropout, device).to(device)
    dataset = load_dataset("loghub/Linux/Linux_2k.log")
    results, template_miner = parse_dataset(dataset)
    log_template_groups = group_parameters(results)
    log_template_tokens, para_tokens = parameter_encoder.tokenize(log_template_groups, device)
    parameter_representation = parameter_encoder(log_template_tokens, para_tokens)
    
    print(parameter_representation.shape)