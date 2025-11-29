# %matplotlib notebook
import numpy as np
import pandas as pd
import random
from config import get_config
import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from model import get_params, make_net


args = get_config()
device = torch.device('cuda:' + args.device)


def main():
    args = get_config()

    # Fix the random seed
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
    print("========================================")
    
    time_max_scale = args.max_scale
    sample_num = 10
    data_name = args.data_name
    block_len = 500
    interp_block_num = args.block_num

    dataset = pd.read_csv(f'../dataset/{data_name}_period_{sample_num}.csv')
    time_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
    df_time = dataset[time_columns] * time_max_scale
    total_time = df_time.values.astype(np.float32)
    sample_test_mse_interps, sample_test_mse_extraps, sample_best_epochs = [], [], []
    sample_test_mae_interps, sample_test_mae_extraps = [], []
    
    epoch = 2000
    
    model_name=args.model_name
    hidden_dim=args.hidden_dim
    enc_dim=args.enc_dim
    sine_dim=args.sine_dim
    learn_freq_init=args.learn_freq
    inner_freq=args.inner_freq
    time_emb_dim=args.time_emb_dim
    f_emb_dim=args.f_emb_dim
    sine_emb_dim=args.sine_emb_dim
    feature_num=1
    lr=args.lr

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
    
    print("========================================\n")
    print("=============[Model Info]===============\n")
    print(f"# Params : {get_params(net)}")
    print(net)
    print("========================================\n")
    
    for sample_idx in range(sample_num):
        print(f"\n######## Sample {sample_idx} ########")
        
        ### sample data ###
        x_vals = dataset.values[:, sample_idx].astype(np.float32)
        nonzero_idx = x_vals != 0
        
        x_vals = x_vals[nonzero_idx]
        t_vals = total_time[nonzero_idx]
        
        scaler = MinMaxScaler()
        x_vals = x_vals.reshape(-1, 1)
        x_vals = scaler.fit_transform(x_vals)
        
        total_num = t_vals.shape[0]
        
        indices = list(range(0, total_num))
        valid_indices = indices[int(total_num/2):int(total_num/2)+block_len]
        if interp_block_num == 1:
            test_interp_indices = indices[int(total_num/8):int(total_num/8)+block_len]
        elif interp_block_num == 2:
            test_interp_indices = indices[int(total_num/8):int(total_num/8)+block_len]
            test_interp_indices.extend(indices[int(total_num/4*3):int(total_num/4*3)+block_len])
        elif interp_block_num == 3:
            test_interp_indices = indices[int(total_num/8):int(total_num/8)+block_len]
            test_interp_indices.extend(indices[int(total_num/8*3):int(total_num/8*3)+block_len])
            test_interp_indices.extend(indices[int(total_num/4*3):int(total_num/4*3)+block_len])
        test_extrap_indices = indices[-block_len*interp_block_num:]
        
        valid_test_indices = valid_indices.copy()
        valid_test_indices.extend(test_interp_indices)
        valid_test_indices.extend(test_extrap_indices)
        
        valid_test_indices, indices = set(valid_test_indices), set(indices)
        train_indices = indices - valid_test_indices
        train_indices, valid_test_indices, indices = list(train_indices), list(valid_test_indices), list(indices)
        
        print(len(indices),len(train_indices),len(valid_indices),len(test_interp_indices), len(test_extrap_indices))
        
        train_x, train_t = x_vals[train_indices].reshape(-1, 1), t_vals[train_indices]
        valid_x, valid_t = x_vals[valid_indices].reshape(-1, 1), t_vals[valid_indices]
        test_interp_x, test_interp_t = x_vals[test_interp_indices].reshape(-1, 1), t_vals[test_interp_indices]
        test_extrap_x, test_extrap_t = x_vals[test_extrap_indices].reshape(-1, 1), t_vals[test_extrap_indices]
        
        
        x_train = Variable(torch.from_numpy(train_x), requires_grad=False).to(device)
        x_train = x_train.to(torch.float32)
        t_train = Variable(torch.from_numpy(train_t), requires_grad=False).to(device)
        t_train = t_train.to(torch.float32)
        
        x_valid = Variable(torch.from_numpy(valid_x), requires_grad=False).to(device)
        x_valid = x_valid.to(torch.float32)
        t_valid = Variable(torch.from_numpy(valid_t), requires_grad=False).to(device)
        t_valid = t_valid.to(torch.float32)
        
        x_test_interp = Variable(torch.from_numpy(test_interp_x), requires_grad=False).to(device)
        x_test_interp = x_test_interp.to(torch.float32)
        t_test_interp = Variable(torch.from_numpy(test_interp_t), requires_grad=False).to(device)
        t_test_interp = t_test_interp.to(torch.float32)
        
        x_test_extrap = Variable(torch.from_numpy(test_extrap_x), requires_grad=False).to(device)
        x_test_extrap = x_test_extrap.to(torch.float32)
        t_test_extrap = Variable(torch.from_numpy(test_extrap_t), requires_grad=False).to(device)
        t_test_extrap = t_test_extrap.to(torch.float32)

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
        
        mse_cost_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        best_epoch = 0
        best_valid_mse = 1000000
        best_test_mse_interp = 1000000
        best_test_mse_extrap = 1000000
        best_test_mae_interp = 1000000
        best_test_mae_extrap = 1000000
        
        for ep in range(1, epoch + 1):
            net.train()
            optimizer.zero_grad()

            pred_data = net(t_train)
            cost = mse_cost_function(pred_data, x_train)
            
            cost.backward()
            optimizer.step()

            with torch.autograd.no_grad():
                net.eval()
                
                pred_data_val = net(t_valid)
                valid_mse = mse_cost_function(pred_data_val, x_valid)
                valid_mse = valid_mse.item()
                
                pred_data_test_interp = net(t_test_interp)
                test_mse_interp = mse_cost_function(pred_data_test_interp, x_test_interp)
                test_mse_interp = test_mse_interp.item()
                test_mae_interp = torch.mean(torch.abs(pred_data_test_interp - x_test_interp))
                test_mae_interp = test_mae_interp.item()
                
                pred_data_test_extrap = net(t_test_extrap)
                test_mse_extrap = mse_cost_function(pred_data_test_extrap, x_test_extrap)
                test_mse_extrap = test_mse_extrap.item()
                test_mae_extrap = torch.mean(torch.abs(pred_data_test_extrap - x_test_extrap))
                test_mae_extrap = test_mae_extrap.item()
                
                if best_valid_mse > valid_mse:
                    best_valid_mse = valid_mse
                    best_epoch = ep
                    best_test_mse_interp = test_mse_interp
                    best_test_mse_extrap = test_mse_extrap
                    best_test_mae_interp = test_mae_interp
                    best_test_mae_extrap = test_mae_extrap
                    
                
                if ep % 100 == 0:                        
                    print(f'{ep}, train : {cost.item():.5f}, valid :, {valid_mse:.5f}, test (interp) :, {test_mse_interp:.5f}, test (extrap) :, {test_mse_extrap:.5f}')

                    
        sample_test_mse_interps.append(best_test_mse_interp)
        sample_test_mse_extraps.append(best_test_mse_extrap)
        sample_test_mae_interps.append(best_test_mae_interp)
        sample_test_mae_extraps.append(best_test_mae_extrap)       
        
        sample_best_epochs.append(best_epoch)
        print(f"- Best Test MSE (INTERP): {best_test_mse_interp} | (EXTRAP) : {best_test_mse_extrap} (At epoch {best_epoch})")
        print(f"- Best Test MAE (INTERP): {best_test_mae_interp} | (EXTRAP) : {best_test_mae_extrap} (At epoch {best_epoch})")
    
    best_interp_avg = np.mean(sample_test_mse_interps)
    best_extrap_avg = np.mean(sample_test_mse_extraps)
    best_interp_avg_mae = np.mean(sample_test_mae_interps)
    best_extrap_avg_mae = np.mean(sample_test_mae_extraps)
    
    print(f"\nBest MSE List (Interp): {sample_test_mse_interps}\nBest MSE List (Extrap): {sample_test_mse_extraps}")
    print(f"\nBest MSE Average (Interp) : {best_interp_avg} | (Extrap) : {best_extrap_avg}")
    print(f"\nBest MAE List (Interp): {sample_test_mae_interps}\nBest MSE List (Extrap): {sample_test_mae_extraps}")
    print(f"\nBest MAE Average (Interp) : {best_interp_avg_mae} | (Extrap) : {best_extrap_avg_mae}")    
    

if __name__ == "__main__":
    main()
