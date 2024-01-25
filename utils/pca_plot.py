import numpy as np
import torch.backends.cudnn as cudnn
import torch

import argparse
import logging
import time
import os
import sys

from architecture.scPSAD import SCPSAD
from architecture.preprocessing_scheme import PreprocessingScheme

import warnings
warnings.filterwarnings("ignore")

import random
import tensorflow as tf

parser = argparse.ArgumentParser('single-cell Preprocessing Schemes Auto-Discovery')
# Data
parser.add_argument('--Hdf5Path_ref', type=str, nargs='+', default=['/home/hym2/projects_dir/scPSAD_v11/data/Mouse_Ileum/mouse_ileum_trans.h5ad'])
parser.add_argument('--Hdf5Path_target', type=str, default='/home/hym2/projects_dir/scPSAD_v11/data/Mouse_Ileum/mouse_ileum_st.h5ad')
# Policy Net architecture
parser.add_argument('--graph_max_size', type=float, default=10)
parser.add_argument('--embedding_size', type=int, default=16)
parser.add_argument('--hidden_size', type=int, default=50)
# Rewarder
parser.add_argument('--task', type=str
                            , choices=['clustering', 'modality_transfer']
                            , default = 'modality_transfer'
                            , help = 'Clustering: Supervised Transfer Design Strategy; Modality Transfer: Supervised Customized')
parser.add_argument('--model', type=str, choices=['kmeans', 'gmm', 'louvain'], default='louvain')
parser.add_argument('--resolution', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=200)
# Choice of optimization
parser.add_argument('--optimization', type=str
                            , choices=['rs', 'pg', 'ppo']
                            , default = 'ppo'
                            , help = 'rs: random search; pg: policy gradient; ppo: proximal policy optimization')
parser.add_argument('--arch_epochs', type=int, default=50, help = 'Set as 100, 50 and xxx by default for clustering, modality transfer and classification respectively.')
parser.add_argument('--arch_lr', type=float, default=1e-3, help = 'Set as 1e-4, 1e-3 and xxx by default for clustering, modality transfer and classification respectively.')
parser.add_argument('--episodes', type=int, default=20)
parser.add_argument('--entropy_weight', type=float, default=1e-4)
parser.add_argument('--graph_length_factor', type=float, default=0.01)
parser.add_argument('--baseline_weight', type=float, default=0.95)
parser.add_argument('--nan_penalty', type=float, default=-0.1)
# PPO Optimization
parser.add_argument('--ppo_epochs', type=int, default=10)
parser.add_argument('--clip_epsilon', type=float, default=0.2)
# Basic configuration
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--check_point_epoch', type=int, default=10)
parser.add_argument('--seed_search', type=int, default=0, help='random seed for reinforcement learning')
parser.add_argument('--seed_test', type=int, default=0, help='random seed for assessing the preprocessing schemes')
args = parser.parse_args()



def main():
    exp_dir = '{}_search_{}'.format(args.optimization, time.strftime("%Y%m%d-%H%M%S"))
    args.top_5_save_dir = exp_dir + '/top_5_computation_graph/'
    args.top_1_model_save_dir = exp_dir + '/'
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(args.top_5_save_dir):
        os.mkdir(args.top_5_save_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream = sys.stdout, level = logging.INFO,
                        format = log_format, datefmt = '%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('args = %s', args)

    torch.manual_seed(args.seed_search)
    np.random.seed(args.seed_search)
    random.seed(args.seed_search)
    tf.random.set_seed(args.seed_search)
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(str(args.gpu)))
        cudnn.benchmark = True
        cudnn.enable = True
        logging.info('using gpu : {}'.format(args.gpu))
        torch.cuda.manual_seed(args.seed_search)
    else:
        device = torch.device('cpu')
        logging.info('using cpu')

    controller = SCPSAD(args, device=device).to(device)
    controller.load_state_dict(torch.load('/home/hym2/projects_dir/scPSAD_v11/ppo_search_20220315-203300/Top1_model.pth'))
    controller.eval()

    graph_path = '/home/hym2/projects_dir/scPSAD_v11/ppo_search_20220315-203300/top_5_computation_graph/epoch_39_top_0_0.3270893850206732_graph.npy'
    graph = PreprocessingScheme(args, [None, np.load(graph_path, allow_pickle=True)], None)

    actions_p, actions_log_p = controller.get_p(actions_index)



    print(controller)




if __name__ == '__main__':
    main()