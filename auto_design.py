import numpy as np
import torch.backends.cudnn as cudnn
import torch

import argparse
import logging
import time
import os
import sys

from inference.PPO import PPO_

import warnings
warnings.filterwarnings("ignore")

import random
#import tensorflow as tf

parser = argparse.ArgumentParser('Transcript Preprocessing Pipelines Auto-Discovery')
# Data
parser.add_argument('--Hdf5Path_ref', type=str, default='/home/hym/projects_dir/autocope_submit_v2/data/trans_CD8.h5')
parser.add_argument('--Hdf5Path_target', type=str, default='/home/hym/projects_dir/autocope_submit_v2/data/protein_CD8.h5')

# Policy Net architecture
parser.add_argument('--graph_max_size', type=float, default=10)
parser.add_argument('--embedding_size', type=int, default=16)
parser.add_argument('--hidden_size', type=int, default=50)
# Rewarder
parser.add_argument('--task', type=str
                            , choices=['cell_continuous', 'cell_discrete', 'gene_continuous']
                            , default = 'cell_continuous')
parser.add_argument('--regressor', type=str, choices=['KNN', 'EN', 'PLSR'], default='EN')
# Choice of optimization
parser.add_argument('--optimization', type=str, default = 'ppo', help = 'ppo: proximal policy optimization')
parser.add_argument('--arch_epochs', type=int, default=50)
parser.add_argument('--arch_lr', type=float, default=1e-3)
parser.add_argument('--episodes', type=int, default=20)
parser.add_argument('--entropy_weight', type=float, default=1e-4)
parser.add_argument('--graph_length_factor', type=float, default=1e-2)
parser.add_argument('--baseline_weight', type=float, default=0.95)
parser.add_argument('--nan_penalty', type=float, default=-1e-1)
# PPO Optimization
parser.add_argument('--ppo_epochs', type=int, default=10)
parser.add_argument('--clip_epsilon', type=float, default=0.2)
# Basic configuration
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--check_point_epoch', type=int, default=10)
parser.add_argument('--seed_search', type=int, default=0, help='random seed for reinforcement learning')
parser.add_argument('--seed_test', type=int, default=0, help='random seed for assessing the preprocessing pipelines')

args = parser.parse_args()

def main():
    exp_dir = 'search_{}_{}'.format('PPO', time.strftime("%Y%m%d-%H%M%S"))
    args.top_5_save_dir = exp_dir + '/top_5_computation_graph/'
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(args.top_5_save_dir):
        os.mkdir(args.top_5_save_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('args = %s', args)

    torch.manual_seed(args.seed_search)
    np.random.seed(args.seed_search)
    random.seed(args.seed_search)
    #tf.random.set_seed(args.seed_search)
    if torch.cuda.is_available() and False:
        device = torch.device('cuda:{}'.format(str(args.gpu)))
        cudnn.benchmark = True
        cudnn.enable = True
        logging.info('using gpu : {}'.format(args.gpu))
        torch.cuda.manual_seed(args.seed_search)
    else:
        device = torch.device('cpu')
        logging.info('using cpu')

    ppo = PPO_(args, device)
    ppo.multi_solve_environment()


if __name__ == '__main__':
    main()