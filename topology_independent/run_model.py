import argparse
import yaml
from utils.loader import load_data_norm
from utils.metrics import log_metrics

def load_model(config):
    if config["model_name"] == "LSTM":
        from models.lstm_model import LSTMModel
        return LSTMModel(config)
    elif config["model_name"] == "Transformer":
        from models.transformer_model import TransformerModel
        return TransformerModel(config)
    else:
        raise ValueError("Unknown model")

def main(cfg):

    train_loader, val_loader, test_loader,stats = load_data_norm(cfg)

    model = load_model(cfg)
    model.set_denorm_stats(stats)
    model.model_train(train_loader,cfg)
    val_metrics = model.evaluate(val_loader, label="Val")
    test_metrics = model.evaluate(test_loader, label="Test")

    all_metrics = {
        "val_R2": val_metrics["R2"],
        "val_NRMSE": val_metrics["NRMSE"],
        "test_R2": test_metrics["R2"],
        "test_NRMSE": test_metrics["NRMSE"],
    }
    log_metrics(all_metrics,cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML file")
    parser.add_argument("--task", type=str, required=True,
                        choices=["load", "solar", "wind", "lmp"])
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)

    cfg = cfg_yaml["common"] | cfg_yaml["tasks"][args.task]
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    main(cfg)