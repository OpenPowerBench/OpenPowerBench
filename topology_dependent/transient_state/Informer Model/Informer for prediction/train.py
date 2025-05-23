
import torch
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import numpy as np
from .utils import build_informer_inputs

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
