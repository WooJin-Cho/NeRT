import torch
import numpy as np
from config import get_config
import torch.backends.cudnn as cudnn
import random
from model import make_net, get_params
from data_processing import longterm_preprocessor
from torch.utils.data import DataLoader

args = get_config()
device = torch.device('cuda:' + args.device)


def main():
    args = get_config()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    device = torch.device('cuda:' + args.device)
    print("========================================")
    print("Use Device :", device)
    print("Available cuda devices :", torch.cuda.device_count())
    print("Current cuda device :", torch.cuda.current_device())
    print("Name of cuda device :", torch.cuda.get_device_name(device))
    print(args)
    print("========================================")

    model_name=args.model_name
    hidden_dim=args.hidden_dim
    enc_dim=args.enc_dim
    sine_dim=args.sine_dim
    learn_freq_init=args.learn_freq
    inner_freq=args.inner_freq
    time_emb_dim=args.time_emb_dim
    f_emb_dim=args.f_emb_dim
    sine_emb_dim=args.sine_emb_dim
    
    train_rat = args.train_rat
    valid_rat = args.valid_rat
    time_max_scale = args.max_scale
    epoch = 10000 #args.epoch
    lr = args.lr
    ###################################################################
    # Data pre-processing & Data-loader
    data_name = f'{args.data_name}_period'
    data_path = f'./dataset/{data_name}.csv'

    assert (train_rat + 2 * valid_rat) == 1

    train_dataset, valid_dataset, test_dataset, feature_num = longterm_preprocessor(data_path=data_path,
                                                                           time_max_scale=time_max_scale,
                                                                           train_ratio=train_rat,
                                                                           valid_ratio=valid_rat)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=100000)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=100000)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=100000)
    ###################################################################



    net = make_net(model_name=model_name,
                   feature_num=feature_num,
                   hidden_dim=hidden_dim,
                   enc_dim=enc_dim,
                   sine_dim=sine_dim,
                   learn_freq_init=learn_freq_init,
                   inner_freq=inner_freq,
                   time_emb_dim=time_emb_dim,
                   f_emb_dim=f_emb_dim,
                   sine_emb_dim=sine_emb_dim,
                   device=device
                   )

    net = net.to(device)
    net_size = get_params(net)

    print("=============[Train Info]===============")
    print(f"- Model size : {net_size}")
    
    # Information of Training Dataset
    print("========================================\n")
    print("=============[Model Info]===============\n")
    print(net)
    print("========================================\n")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    mse_cost_function = torch.nn.MSELoss()

    valid_mse_lst = []
    test_mse_lst = []
    best_epoch = 0
    best_valid_mse = 1000000
    best_test_mse = 1000000
    best_test_mae = 1000000

    for ep in range(1, epoch + 1):
        net.train()
        optimizer.zero_grad()

        loss_saver = 0
        valid_error_saver = 0
        valid_error_saver_mae = 0
        test_error_saver = 0
        test_error_saver_mae = 0

        for sample_train in train_dataloader:
            time_idx, feature_idx, gt_data = sample_train
            time_idx = time_idx.clone().detach().to(device)
            feature_idx = feature_idx.clone().detach().to(device)
            gt_data = gt_data.clone().detach().to(device)
            pred_data = net(time_idx, feature_idx)
            pred_data = pred_data.squeeze(-1)
            cost = mse_cost_function(pred_data, gt_data)  

            loss_saver += cost.item()
            cost.backward()
            optimizer.step()

        with torch.autograd.no_grad():
            net.eval()
            
            for sample_valid in valid_dataloader:
                time_idx_valid, feature_idx_valid, gt_data_valid = sample_valid
                time_idx_valid = time_idx_valid.clone().detach().to(device)
                feature_idx_valid = feature_idx_valid.clone().detach().to(device)
                gt_data_valid = gt_data_valid.clone().detach().to(device)

                pred_data_valid = net(time_idx_valid, feature_idx_valid)
                pred_data_valid = pred_data_valid.squeeze(-1)
                valid_error = torch.sum((pred_data_valid - gt_data_valid) ** 2)
                valid_error_mae = torch.sum(torch.abs(pred_data_valid - gt_data_valid))

                valid_error_saver += valid_error.item()
                valid_error_saver_mae += valid_error_mae.item()

            valid_error_saver = valid_error_saver / len(valid_dataset)
            valid_error_saver_mae = valid_error_saver_mae / len(valid_dataset)
            
            for sample_test in test_dataloader:
                time_idx_test, feature_idx_test, gt_data_test = sample_test
                time_idx_test = time_idx_test.clone().detach().to(device)
                feature_idx_test = feature_idx_test.clone().detach().to(device)
                gt_data_test = gt_data_test.clone().detach().to(device)

                pred_data_test = net(time_idx_test, feature_idx_test)
                pred_data_test = pred_data_test.squeeze(-1)
                test_error = torch.sum((pred_data_test - gt_data_test) ** 2)
                test_error_mae = torch.sum(torch.abs(pred_data_test - gt_data_test))

                test_error_saver += test_error.item()
                test_error_saver_mae += test_error_mae.item()

            test_error_saver = test_error_saver / len(test_dataset)
            test_error_saver_mae = test_error_saver_mae / len(test_dataset)
            
            valid_mse_lst.append(valid_error_saver)
            test_mse_lst.append(test_error_saver)

            if valid_error_saver < best_valid_mse:
                best_valid_mse = valid_error_saver
                best_epoch = ep
                best_test_mse = test_error_saver
                best_test_mae = test_error_saver_mae
            
            if ep == 1:
                print(f'{"EPOCH":>6} : {"TRAIN MSE":>10s} // {"VALID MSE":>10s} / {"TEST MSE":>10s} // {"VALID MAE":>10s} / {"TEST MAE":>10s}')

            if ep % 10 == 0:
                print(f'{ep:>6} : {loss_saver:>10.5f} // {valid_error_saver:>10.5f} / {test_error_saver:>10.5f} // {valid_error_saver_mae:>10.5f} / {test_error_saver_mae:>10.5f}')


    print(f"BEST MSE : {best_test_mse} | BEST MAE : {best_test_mae} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
