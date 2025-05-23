import torch
from torch.utils.data import Dataset

class PredictionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

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
