
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

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
