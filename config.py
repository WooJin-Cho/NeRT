import argparse
import os
import sys
import numpy as np
parser = argparse.ArgumentParser()
# CPU / GPU setting


parser.add_argument('--device', type=str, default='0')
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--model_name', type=str, default='nert_uni')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--sine_dim', type=int, default=30)
parser.add_argument('--learn_freq', type=float, default=10.)
parser.add_argument('--time_emb_dim', type=int, default=30)
parser.add_argument('--inner_freq', type=float, default=1.)
parser.add_argument('--f_emb_dim', type=int, default=30)
parser.add_argument('--sine_emb_dim', type=int, default=30)
parser.add_argument('--enc_dim', type=int, default=20)

parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--max_scale', type=int, default=100)
parser.add_argument('--data_name', type=str, default='depts_traffic')
parser.add_argument('--train_rat', type=float, default=0.7)
parser.add_argument('--valid_rat', type=float, default=0.15)
parser.add_argument('--block_num', type=int, default=1)


def get_config():
    return parser.parse_args()
