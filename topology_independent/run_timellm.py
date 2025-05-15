import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import TimeLLM
from sklearn.metrics import mean_squared_error,r2_score

from utils.loader import load_data
import time
import numpy as np
import os
from utils.metrics import log_metrics_timellm

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon Metal (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))

def vali(args, model, vali_data, vali_loader, criterion):
    total_loss = []
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader)):
            if i ==1:
                break
            batch_x = batch_x.unsqueeze(-1).float().to(device)
            batch_y = batch_y.unsqueeze(-1).float()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x)[0]
                    else:
                        outputs = model(batch_x)
            else:
                if args.output_attention:
                    outputs = model(batch_x)[0]
                else:
                    outputs = model(batch_x)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach()
            true = batch_y.detach()
            y_true.append(true.squeeze(-1))
            y_pred.append(pred.squeeze(-1))

            loss = criterion(pred, true)

            total_loss.append(loss.item())


    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    nrmse = 100 * (np.sqrt(mean_squared_error(y_true, y_pred)) / (np.average(y_true)))
    r2 = r2_score(y_true, y_pred)

    model.train()
    return total_loss, nrmse, r2

def main():
    for ii in range(args.itr):
        train_data, vali_data, test_data,train_loader,vali_loader,test_loader = load_data(args)
        model = TimeLLM.Model(args).float()

        time_now = time.time()

        train_steps = len(train_loader)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

        criterion = nn.MSELoss()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
                if i==1:
                    break
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().unsqueeze(-1).to(device)
                batch_y = batch_y.float().unsqueeze(-1).to(device)
                model.to(device)
                outputs = model(batch_x)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss , vali_nrmse_loss, vali_r2_loss= vali(args, model, vali_data, vali_loader, criterion)
            test_loss,  test_nrmse_loss, test_r2_loss = vali(args, model, test_data, test_loader, criterion)
                # print(args.task_name)
            print("Epoch: {0} | Train Loss: {1:.7f}".format(epoch + 1, train_loss))
            print("Vali NRMSE:",vali_nrmse_loss)
            print("Vali R2:",vali_r2_loss)
            print("Test NRMSE:",test_nrmse_loss)
            print("Test R2:",test_r2_loss)


            all_metrics = {
                "val_R2": vali_r2_loss,
                "val_NRMSE": vali_nrmse_loss,
                "test_R2": test_r2_loss,
                "test_NRMSE": test_nrmse_loss,
            }
            log_metrics_timellm(all_metrics, args)
            if epoch == 0:
                args.learning_rate = model_optim.param_groups[0]['lr']
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=True)

if __name__ == '__main__':
    import argparse, yaml, pathlib
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

    parser = argparse.ArgumentParser(description='Time-LLM')
    parser.add_argument(
        "--config", type=str,
        help="Path to YAML file whose keys mirror command-line flags."
    )

    # basic config
    parser.add_argument('--task_name', type=str,
                        help='task name, options:[load, wind, solar, lmp]')
    parser.add_argument('--model_name', type=str, default='TimeLLM',
                        help='model name')

    # data loader
    parser.add_argument('--x_path', type=str, help='training data path')
    parser.add_argument('--y_path', type=str, help='training label path')
    parser.add_argument('--log_path', type=str, help='log path')

    # forecasting task
    parser.add_argument('--seq_len', type=int, help='input sequence length')
    parser.add_argument('--pred_len', type=int, help='prediction sequence length')


    # model define
    parser.add_argument('--enc_in', type=int, help='encoder input size')
    parser.add_argument('--dec_in', type=int, help='decoder input size')
    parser.add_argument('--c_out', type=int, help='output size')
    parser.add_argument('--d_model', type=int, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')  # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='2048',
                        help='LLM model dimension')  # LLama7b:4096; Llama1b:2048
    parser.add_argument('--instruction', type=str, help='instruction')

    # optimization
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int)
    parser.add_argument('--percent', type=int, default=100)

    cli = parser.parse_args()

    yaml_dict = {}
    if cli.config:
        yaml_path = pathlib.Path(cli.config)
        yaml_dict = yaml.safe_load(yaml_path.read_text())

    merged = {**yaml_dict, **{k: v for k, v in vars(cli).items() if v is not None}}
    args = argparse.Namespace(**merged)

    main()