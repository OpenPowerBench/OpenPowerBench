import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.base_model import BaseModel
from sklearn.metrics import mean_squared_error, r2_score

class LSTMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.input_dim = 1
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = 1
        self.epochs = config["epochs"]
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True
        )

        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

        self.model = nn.Sequential(self.lstm, self.linear)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.5
        )
    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out)
        return out.squeeze(-1)

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
            print(f"[LSTM] Epoch {epoch+1}/{self.epochs} - Loss: {total_loss:.4f}")
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

        nrmse = np.sqrt(mean_squared_error(y_true, y_pred))/self.y_mean*100
        r2 = r2_score(y_true, y_pred)
        print(f"[{label}] R2: {r2:.4f}, NRMSE: {nrmse:.4f}")
        return {"R2": r2, "NRMSE": nrmse}

    def save(self, path):
        torch.save(self, path)
        print(f"[LSTM] Full model saved to {path}")

    @staticmethod
    def load(path, device=None):
        model = torch.load(path, map_location=device or torch.device("cpu"))
        print(f"[LSTM] Full model loaded from {path}")
        return model
