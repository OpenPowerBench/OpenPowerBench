
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.informer import Informer
from .utils.dataset import ClassificationDataset, load_and_prepare_data
from .model import InformerClassifier
from .train import train_classifier, evaluate_classifier

data_path = "/IEEE14_data.pckl"
label_path = "/IEEE14_label.pckl"

label_target = 'location' #'type' or 'location'
X, y, n_classes, input_dim = load_and_prepare_data(data_path, label_path, label_target)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1667, random_state=42, stratify=y_temp)

train_loader = DataLoader(ClassificationDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)), batch_size=32, shuffle=True)
val_loader = DataLoader(ClassificationDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val)), batch_size=32, shuffle=False)
test_loader = DataLoader(ClassificationDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)), batch_size=32, shuffle=False)

d_model = 512
seq_len = X.shape[1]

informer = Informer(enc_in=input_dim, dec_in=input_dim, c_out=input_dim, seq_len=seq_len, label_len=60, out_len=0, d_model=d_model).cuda()
model = InformerClassifier(informer, d_model=d_model, n_classes=n_classes, seq_len=seq_len).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    loss = train_classifier(model, train_loader, optimizer, criterion)
    train_acc, _ = evaluate_classifier(model, train_loader)
    val_acc, _ = evaluate_classifier(model, val_loader)
    test_acc, _ = evaluate_classifier(model, test_loader)
    print(f"[{label_target.upper()}] Epoch {epoch} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
