
import torch.nn as nn

class InformerClassifier(nn.Module):
    def __init__(self, informer_model, d_model, n_classes, seq_len):
        super().__init__()
        self.embedding = informer_model.enc_embedding
        self.encoder = informer_model.encoder
        self.seq_len = seq_len
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x_mark = torch.zeros(B, self.seq_len, 4).to(x.device)
        x = self.embedding(x, x_mark)
        out, _ = self.encoder(x)
        out = self.classifier(out.permute(0, 2, 1))
        return out
