import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
from models.informer import Informer
import pickle
from sklearn.model_selection import train_test_split

class PredictionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X  # shape: (num_samples, 121, N)
        self.Y = Y  # shape: (num_samples, 122, N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def build_informer_inputs(x, y, label_len, pred_len):
    B, seq_len, C = x.shape
    x_mark_enc = torch.zeros((B, seq_len, 4)).to(x.device)
    x_dec = torch.cat([y[:, :label_len, :], torch.zeros((B, pred_len - label_len, C)).to(x.device)], dim=1)
    x_mark_dec = torch.zeros((B, pred_len, 4)).to(x.device)
    return x, x_mark_enc, x_dec, x_mark_dec

def train_predictor(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        x_enc, x_mark_enc, x_dec, x_mark_dec = build_informer_inputs(x, y, label_len=60, pred_len=122)
        optimizer.zero_grad()
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_predictor(model, dataloader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            x_enc, x_mark_enc, x_dec, x_mark_dec = build_informer_inputs(x, y, label_len=60, pred_len=122)
            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec).cpu().numpy()
            preds.append(pred)
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mape = mean_absolute_percentage_error(trues.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    r2 = r2_score(trues.flatten(), preds.flatten())
    nrmse = rmse / np.mean(np.abs(trues))
    return mape, r2, rmse, nrmse



data_path = r"/IEEE118_pred.pckl"

with open(data_path, 'rb') as f:
    pred_data = pickle.load(f) 

pred_data = np.array(pred_data)
input_dim = pred_data.shape[1]
X_pred = pred_data[:, :, :121]
Y_pred = pred_data[:, :, 121:]
X_pred = np.transpose(X_pred, (0, 2, 1))  # (N, 121, N_feature)
Y_pred = np.transpose(Y_pred, (0, 2, 1))  # (N, 122, N_feature)

# Split dataset
X_temp, X_test, Y_temp, Y_test = train_test_split(X_pred, Y_pred, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.1667, random_state=42)

# DataLoaders
batch_size = 32
train_loader = DataLoader(PredictionDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(PredictionDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(PredictionDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

d_model = 512
seq_len = 121
label_len = 60
pred_len = 122

pred_model = Informer(
    enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
    seq_len=seq_len, label_len=label_len, out_len=pred_len, d_model=d_model
).cuda()

pred_optim = torch.optim.Adam(pred_model.parameters(), lr=1e-4)
pred_criterion = torch.nn.MSELoss()

for epoch in range(20):
    train_loss = train_predictor(pred_model, train_loader, pred_optim, pred_criterion)
    val_mape, val_r2, val_rmse, val_nrmse = evaluate_predictor(pred_model, val_loader)
    test_mape, test_r2, test_rmse, test_nrmse = evaluate_predictor(pred_model, test_loader)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
          f"Val -> MAPE: {val_mape:.4f}, R2: {val_r2:.4f}, NRMSE: {val_nrmse:.4f} | "
          f"Test -> MAPE: {test_mape:.4f}, R2: {test_r2:.4f}, NRMSE: {test_nrmse:.4f}")
