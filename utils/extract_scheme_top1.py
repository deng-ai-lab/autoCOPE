import argparse
import os
from extractor_scheme_top1 import Extractor, save_extracted_dataset

#Setting 1 GPU available for me.
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'


def extracting(args):
    extractor = Extractor(args.Hdf5Path_ref, args.graph_path, args.seed_test, args)
    save_extracted_dataset(extractor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Graph Reinforcement Learning for Task-Specific Preprocess Discovery')
    # task
    parser.add_argument('--graph_path', type=str, default='')
    # data
    parser.add_argument('--Hdf5Path_ref', type=str, default='')
    parser.add_argument('--Hdf5Path_target', type=str, default='')
    parser.add_argument('--save_path', type=str, default="")

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--seed_test', type=int, default=0, help='random seed for assessing the preprocessing schemes')
    args = parser.parse_args()

    print('Graph Path: ' + args.graph_path)
    extracting(args)