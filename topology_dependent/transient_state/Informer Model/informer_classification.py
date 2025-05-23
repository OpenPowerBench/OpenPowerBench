import pickle
import numpy as np
from itertools import product
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from models.informer import Informer
from collections import OrderedDict
import os
from sklearn.model_selection import train_test_split



class ClassificationDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # shape: (num_samples, 243, N)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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

def train_classifier(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_classifier(model, dataloader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            pred = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(y.cpu().numpy())
    acc = accuracy_score(trues, preds)
    bal_acc = balanced_accuracy_score(trues, preds)
    return acc, bal_acc

def load_and_prepare_data(ts_file, label_file, label_target='type'):
    with open(ts_file, 'rb') as f:
        ts_data = pickle.load(f)  # (N, feature_dim, T)

    n_bus = ts_data.shape[1]

    with open(label_file, 'rb') as f:
        label_dicts_raw = pickle.load(f)
    
    print(f"label list length: {len(label_dicts_raw)}, label[0]:{label_dicts_raw[0]}")

    label_dicts = []
    for entry in label_dicts_raw:
        clean_entry = {k: v.iloc[0] if hasattr(v, 'iloc') else v for k, v in entry.items()}
        label_dicts.append(clean_entry)

    print(f"label list length: {len(label_dicts)}, label[0]:{label_dicts[0]}")

    ts_data = np.transpose(ts_data, (0, 2, 1))  # (N, T, feature_dim)

    # type label mapping
    type_set = sorted(list(set(d['type'] for d in label_dicts)))
    type_map = {k: i for i, k in enumerate(type_set)}

    # location label mapping
    buses = list(range(1, n_bus+1))
    ground_pairs = [f"{bus}_-1" for bus in buses]

    location_pairs = set()
    for d in label_dicts:
        b1, b2 = d['bus1'], d['bus2']
        key = f"{b1}_{b2}"
        location_pairs.add(key)

    full_location_set = sorted(location_pairs.union(ground_pairs))
    location_map = {name: idx for idx, name in enumerate(full_location_set)}

    print(f"full_location_set:{full_location_set}")
    print(f"Total locations: {len(location_map)}")
    print(f"location_map: {location_map}")

    # Create label vector
    if label_target == 'type':
        labels = np.array([type_map[d['type']] for d in label_dicts])
    elif label_target == 'location':
        labels = []
        for d in label_dicts:
            key1 = f"{d['bus1']}_{d['bus2']}"
            key2 = f"{d['bus2']}_{d['bus1']}"
            if key1 in location_map:
                labels.append(location_map[key1])
            elif key2 in location_map:
                labels.append(location_map[key2])
    else:
        raise ValueError("Unknown label target")

    print(f"[INFO] Final shape: X = {ts_data.shape}, y = {np.array(labels).shape}")
    return ts_data, labels, len(type_map) if label_target == 'type' else len(location_map), ts_data.shape[2]



# ----- Main Pipeline -----
data_path = r"/IEEE118_data.pckl"
label_path = r"/IEEE118_label.pckl"


label_target = 'location'  # 'type' or 'location'
X, y, n_classes, input_dim = load_and_prepare_data(data_path, label_path, label_target=label_target)
print(f"X shape: {X.shape}, y shape: {len(y)},n_classes:{n_classes}, input_dim:{input_dim}")

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1667, random_state=42, stratify=y_temp)

train_dataset = ClassificationDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = ClassificationDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = ClassificationDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

d_model = 512
informer_model = Informer(enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
                          seq_len=X.shape[1], label_len=60, out_len=0, d_model=d_model).cuda()

model = InformerClassifier(informer_model, d_model=d_model, n_classes=n_classes, seq_len=X.shape[1]).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    loss = train_classifier(model, train_loader, optimizer, criterion)
    train_acc, train_bal_acc = evaluate_classifier(model, train_loader)
    val_acc, val_bal_acc = evaluate_classifier(model, val_loader)
    test_acc, test_bal_acc = evaluate_classifier(model, test_loader)

    print(f"[{label_target.upper()}] Epoch {epoch} | "
          f"Loss: {loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Test Acc: {test_acc:.4f}")

