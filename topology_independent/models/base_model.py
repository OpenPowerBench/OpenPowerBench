import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def model_train(self, train_data,config):
        raise NotImplementedError

    def evaluate(self, val_data):
        raise NotImplementedError