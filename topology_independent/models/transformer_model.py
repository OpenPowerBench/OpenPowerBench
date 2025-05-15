import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from models.base_model import BaseModel


class TransformerModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_dim = 1
        self.embed_dim = config["hidden_dim"]
        self.output_dim = 1
        self.seq_len = config.get("seq_len", 24)
        self.epochs = config["epochs"]
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config["num_heads"],
            dim_feedforward=config.get("ff_dim", self.embed_dim * 4),
            dropout=config.get("dropout", 0.1),
            batch_first=True  # for (B, T, D) format
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.output_proj = nn.Linear(self.embed_dim, self.output_dim)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.5
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)

        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        out = self.output_proj(x).squeeze(-1)

        return out

    def model_train(self, train_loader,config):
        self.to(self.device)
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for x, y in train_loader:
                x = x.float().to(self.device)
                y = y.float().to(self.device)

                self.optimizer.zero_grad()
                output = self.forward(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            print(f"[Transformer] Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")
        self.save(config["model_path"])

    def set_denorm_stats(self, stats):
        self.y_mean = stats["y_mean"]
        self.y_std = stats["y_std"]

    def denormalize(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    def evaluate(self, data_loader, label="Val"):
        self.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.float().to(self.device)
                output = self.forward(x)
                y_true.append(y)
                y_pred.append(output)

        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        y_true = self.denormalize(torch.tensor(y_true)).numpy()
        y_pred = self.denormalize(torch.tensor(y_pred)).numpy()

        nrmse = 100*(np.sqrt(mean_squared_error(y_true, y_pred))/(np.average(y_true)))
        r2 = r2_score(y_true, y_pred)
        print(f"[{label}] R2: {r2:.4f}, NRMSE: {nrmse:.4f}")
        return {"R2": r2, "NRMSE": nrmse}

    def set_denorm_stats(self, stats):
        self.y_mean = stats["y_mean"]
        self.y_std = stats["y_std"]

    def denormalize(self, y_norm):
        return y_norm * self.y_std + self.y_mean

    def save(self, path):
        torch.save(self, path)
        print(f"[Transformer] Full model saved to {path}")

    @staticmethod
    def load(path, device=None):
        model = torch.load(path, map_location=device or torch.device("cpu"))
        print(f"[Transformer] Full model loaded from {path}")
        return model
