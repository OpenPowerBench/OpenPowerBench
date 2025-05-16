import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def _split_dataset(x_path, y_path, split=(0.7, 0.15, 0.15), seed=42):
    x = torch.from_numpy(np.load(x_path))
    y = torch.from_numpy(np.load(y_path))

    dataset = TensorDataset(x, y)
    n = len(dataset)
    n_train = int(split[0] * n)
    n_val   = int(split[1] * n)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator)


def _z_score(tensor, mean, std):
    return (tensor - mean) / (std + 1e-8)


def load_data_norm(config):
    train_set, val_set, test_set = _split_dataset(config['x_path'], config['y_path'])
    batch_size = 32

    # Compute stats on training data only
    x_train = torch.stack([s[0] for s in train_set])
    y_train = torch.stack([s[1] for s in train_set])
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    def _apply_norm(subset):
        xs = torch.stack([_z_score(s[0], x_mean, x_std) for s in subset])
        ys = torch.stack([_z_score(s[1], y_mean, y_std) for s in subset])
        return TensorDataset(xs, ys)

    norm_train = _apply_norm(train_set)
    norm_val   = _apply_norm(val_set)
    norm_test  = _apply_norm(test_set)

    stats = {"y_mean": y_mean.item(), "y_std": y_std.item()}

    return (
        DataLoader(norm_train, batch_size=batch_size, shuffle=True),
        DataLoader(norm_val,   batch_size=batch_size),
        DataLoader(norm_test,  batch_size=batch_size),
        stats,
    )


def load_data(args):
    train_set, val_set, test_set = _split_dataset(args.x_path, args.y_path)
    batch_size = 32

    return (
        train_set,
        val_set,
        test_set,
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set,   batch_size=batch_size),
        DataLoader(test_set,  batch_size=batch_size),
    )
