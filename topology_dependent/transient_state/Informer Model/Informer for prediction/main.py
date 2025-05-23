
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.informer import Informer
from .utils import PredictionDataset
from train import train_predictor, evaluate_predictor

with open("/IEEE14_pred.pckl", 'rb') as f:
    pred_data = pickle.load(f)

pred_data = np.array(pred_data)
input_dim = pred_data.shape[1]
X_pred = np.transpose(pred_data[:, :, :121], (0, 2, 1))
Y_pred = np.transpose(pred_data[:, :, 121:], (0, 2, 1))

X_temp, X_test, Y_temp, Y_test = train_test_split(X_pred, Y_pred, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.1667, random_state=42)

train_loader = DataLoader(PredictionDataset(X_train, Y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(PredictionDataset(X_val, Y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(PredictionDataset(X_test, Y_test), batch_size=32, shuffle=False)

model = Informer(enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
                 seq_len=121, label_len=60, out_len=122, d_model=512).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

for epoch in range(20):
    train_loss = train_predictor(model, train_loader, optimizer, criterion)
    val_mape, val_r2, val_rmse, val_nrmse = evaluate_predictor(model, val_loader)
    test_mape, test_r2, test_rmse, test_nrmse = evaluate_predictor(model, test_loader)

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
          f"Val -> MAPE: {val_mape:.4f}, R2: {val_r2:.4f}, NRMSE: {val_nrmse:.4f} | "
          f"Test -> MAPE: {test_mape:.4f}, R2: {test_r2:.4f}, NRMSE: {test_nrmse:.4f}")
