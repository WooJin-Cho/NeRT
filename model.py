import torch
import torch.nn as nn

def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        if p.requires_grad == True:
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
    return pp

class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
    

def make_net(model_name, feature_num, hidden_dim, enc_dim, sine_dim, learn_freq_init, inner_freq, time_emb_dim, f_emb_dim, sine_emb_dim, device=None):
    dict_net = {
        'nert_uni': NeRT_univariate(feature_num, hidden_dim, enc_dim, sine_dim, learn_freq_init, inner_freq, time_emb_dim, f_emb_dim, sine_emb_dim),
        'nert_multi': NeRT_multivariate(feature_num, hidden_dim, enc_dim, sine_dim, learn_freq_init, inner_freq, time_emb_dim, f_emb_dim, sine_emb_dim), 
    }
    net = dict_net[model_name]
    return net

 
class learnable_fourier_mapping(nn.Module):
    def __init__(self, w0=25., dim=30):
        super().__init__()
        
        self.amp_year = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.amp_month = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.amp_day = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.amp_hour = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.amp_minute = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.amp_second = nn.Parameter(torch.ones(dim), requires_grad=True)
        
        self.shift_year = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.shift_month = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.shift_day = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.shift_hour = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.shift_minute = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.shift_second = nn.Parameter(torch.zeros(dim), requires_grad=True)
        
        self.freq_year = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.freq_month = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.freq_day = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.freq_hour = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.freq_minute = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        self.freq_second = nn.Parameter(torch.Tensor(1, dim), requires_grad=True)
        
        nn.init.uniform_(self.freq_year, 0., w0)
        nn.init.uniform_(self.freq_month, 0., w0)
        nn.init.uniform_(self.freq_day, 0., w0)
        nn.init.uniform_(self.freq_hour, 0., w0)
        nn.init.uniform_(self.freq_minute, 0., w0)
        nn.init.uniform_(self.freq_second, 0., w0)
        

    def forward(self, x):
        year, month, day, hour, minute, second = x[:,0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], x[:, 4:5], x[:, 5:6]
        sine_year = self.amp_year * torch.sin(self.freq_year * year) + self.shift_year
        sine_month = self.amp_month * torch.sin(self.freq_month * month) + self.shift_month
        sine_day = self.amp_day * torch.sin(self.freq_day * day) + self.shift_day
        sine_hour = self.amp_hour * torch.sin(self.freq_hour * hour) + self.shift_hour
        sine_minute = self.amp_minute * torch.sin(self.freq_minute * minute) + self.shift_minute
        sine_second = self.amp_second * torch.sin(self.freq_second * second) + self.shift_second
        
        sine_time = torch.cat([sine_year, sine_month, sine_day, sine_hour, sine_minute, sine_second], axis = 1)

        return sine_time
    

class NeRT_univariate(nn.Module):
    def __init__(self, feature_num, hidden_dim, enc_dim, sine_dim, learn_freq_init, inner_freq, time_emb_dim, f_emb_dim, sine_emb_dim):
        super(NeRT_univariate, self).__init__()
        
        self.sine = Sine(inner_freq)
        self.learnable_sine = learnable_fourier_mapping(learn_freq_init, sine_dim)
        self.relu = nn.ReLU()
        
        self.enc_time_1 = nn.Linear(6, enc_dim)
        self.enc_time_2 = nn.Linear(enc_dim, time_emb_dim)
        
        self.enc_sine = nn.Linear(sine_dim * 6, sine_emb_dim)
        
        self.enc_scale_1 = nn.Linear(sine_emb_dim+time_emb_dim, enc_dim)
        self.enc_scale_2 = nn.Linear(enc_dim, 1)
        
        self.enc_1 = nn.Linear(sine_emb_dim, hidden_dim)
        self.enc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_3 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_4 = nn.Linear(hidden_dim, hidden_dim) 
        self.enc_5 = nn.Linear(hidden_dim, 1)


        
    def forward(self, time_index):
        encoded_sine = self.enc_sine(self.learnable_sine(time_index))
        encoded_time = self.enc_time_2(self.relu(self.enc_time_1(time_index)))
        
        concat_time = torch.cat([encoded_sine, encoded_time], axis = 1)
        
        emb_scale_out = self.relu(self.enc_scale_1(concat_time))
        emb_scale_out = self.enc_scale_2(emb_scale_out)
        
        emb_out_1 = self.sine(self.enc_1(encoded_sine))
        emb_out_2 = self.sine(self.enc_2(emb_out_1))
        emb_out_3 = self.sine(self.enc_3(emb_out_2))
        emb_out_4 = self.sine(self.enc_4(emb_out_3)) 
        emb_out_5 = self.sine(self.enc_5(emb_out_4)) * emb_scale_out

        return emb_out_5
    


    
        

class NeRT_multivariate(nn.Module):
    def __init__(self, feature_num, hidden_dim, enc_dim, sine_dim, learn_freq_init, inner_freq, time_emb_dim, f_emb_dim, sine_emb_dim):
        super(NeRT_multivariate, self).__init__()
        
        self.sine = Sine(inner_freq)
        self.learnable_sine = learnable_fourier_mapping(learn_freq_init, sine_dim)
        self.relu = nn.ReLU()
        
        self.feature_enc_1 = nn.Linear(feature_num, enc_dim)
        self.feature_enc_2 = nn.Linear(enc_dim, f_emb_dim)
        
        self.enc_time_1 = nn.Linear(6, enc_dim)
        self.enc_time_2 = nn.Linear(enc_dim, time_emb_dim)
        
        self.enc_sine = nn.Linear(sine_dim * 6, sine_emb_dim)
        
        self.enc_scale_1 = nn.Linear(sine_emb_dim+time_emb_dim+f_emb_dim, enc_dim)
        self.enc_scale_2 = nn.Linear(enc_dim, 1)
        
        self.enc_1 = nn.Linear(sine_emb_dim+f_emb_dim, hidden_dim)
        self.enc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_3 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_4 = nn.Linear(hidden_dim, hidden_dim) 
        self.enc_5 = nn.Linear(hidden_dim, 1)


        
    def forward(self, time_index, feature_index):
        encoded_sine = self.enc_sine(self.learnable_sine(time_index))
        encoded_time = self.relu(self.enc_time_1(time_index))
        encoded_time = self.enc_time_2(encoded_time)
        
        encoded_feature = self.relu(self.feature_enc_1(feature_index))
        encoded_feature = self.feature_enc_2(encoded_feature)
        
        concat_time = torch.cat([encoded_sine, encoded_time], axis = 1)
        
        time_feature = torch.cat([concat_time, encoded_feature], axis = 1)
        emb_scale_out = self.relu(self.enc_scale_1(time_feature))
        emb_scale_out = self.enc_scale_2(emb_scale_out)
        
        embedded_input = torch.cat([encoded_sine, encoded_feature],axis = 1)
        emb_out_1 = self.sine(self.enc_1(embedded_input))
        emb_out_2 = self.sine(self.enc_2(emb_out_1))
        emb_out_3 = self.sine(self.enc_3(emb_out_2))
        emb_out_4 = self.sine(self.enc_4(emb_out_3)) 
        emb_out_5 = self.sine(self.enc_5(emb_out_4)) * emb_scale_out

        return emb_out_5

