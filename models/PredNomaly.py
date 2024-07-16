import torch
import torch.nn as nn
from models.SingleLogEncoder import SingleLogEncoder
from models.ParameterEncoder import SingleParaEncoder
from datasets.preprocessing import parse_dataset, load_dataset

class LogAnomalyDetector(nn.Module):
    def __init__(self, embedding_dim=768, transformer_layers=2, n_heads=12, dropout=0.1):
        super(LogAnomalyDetector, self).__init__()
        self.log_encoder = SingleLogEncoder(embedding_dim, transformer_layers, n_heads, dropout)
        self.para_encoder = SingleParaEncoder(embedding_dim, transformer_layers, n_heads, dropout)

        self.para_pooling = nn.AdaptiveAvgPool1d(1)

        self.W_alpha = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.v = nn.Parameter(torch.Tensor(embedding_dim, 1))

        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=2*embedding_dim, nhead=n_heads, dropout=dropout)
            for _ in range(transformer_layers)
        ])

        self.embedding_dim = embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(2*embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, log_messages, para_tokens):
        log_representation = [self.log_encoder(log) for log in log_messages]
        log_representation = torch.stack(log_representation).permute(1, 0, 2)

        print(log_representation.shape)

        parameter_representation = []
        for para_token in para_tokens:
            if len(para_token) == 0:
                para_pooled = torch.zeros(1, self.embedding_dim)
            else:
                para_pooled = torch.stack([self.para_encoder(para) for para in para_token])
                
                para_pooled = torch.mean(para_pooled, dim=0, keepdim=True).squeeze(0)
            parameter_representation.append(para_pooled)
        parameter_representation = torch.stack(parameter_representation).permute(1, 0, 2)
        print(parameter_representation.shape)
        combined_representation = torch.cat([log_representation, parameter_representation], dim=2)
        print(combined_representation.shape)

        for layer in self.transformer_encoder:
            combined_representation = layer(combined_representation)

        output = self.classifier(combined_representation)

        return output
    
    def tokenize(self, log_messages):
        log_template_tokens = []
        para_tokens = []
        for i, log_message in enumerate(log_messages):
            log_template = self.log_encoder.tokenize(log_message["template_mined"])
            para_token = [self.para_encoder.tokenize(parameter.value) for parameter in log_message["parameters"]]
            log_template_tokens.append(log_template)
            para_tokens.append(para_token)
        return log_template_tokens, para_tokens

def test_log_anomaly_detector():
    embedding_dim = 768
    transformer_layers = 2
    n_heads = 12
    dropout = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_anomaly_detector = LogAnomalyDetector(embedding_dim, transformer_layers, n_heads, dropout).to(device)
    dataset = load_dataset("loghub/Linux/Linux_2k.log")
    results, log_messages = parse_dataset(dataset)
    log_template_tokens, para_tokens = log_anomaly_detector.tokenize(results)
    output = log_anomaly_detector(log_template_tokens, para_tokens)
    print(output.shape)
