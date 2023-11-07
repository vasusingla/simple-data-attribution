import torch
import numpy as np
import argparse
from train_utils import binary_search
import pickle as pkl
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-idx', type=int, default=0,
                        help='Index of test example for which data support \
                             needs to be estimated.')
    parser.add_argument('--flip-class', action='store_true')
    parser.add_argument('--rank-path', type=str, default='data/logits/average_rank.npy')
    parser.add_argument('--matrix-path', type=str, default='data/topk_train_samples/dmodel_1280.npy')
    parser.add_argument('--results-path', type=str, default='results/dmodel_1280/')
    parser.add_argument('--num-tests', type=int, default=8)
    parser.add_argument('--search-budget', type=int, default=8)
    parser.add_argument('--arch', type=str, default='resnet9')
    return parser.parse_args()

def main(args):
    # args = parse_args()
    test_idxs = np.load('data/test_indices.npy/test_100.npy')
    for idx, test_idx in test_idxs:
        topk_matrix = np.load(args.matrix_path)
        train_idxs = topk_matrix[idx]
        if args.flip_class:
            test_labels = np.load('data/info/test_labels.npy')
            rank_info = np.load(args.rank_path)[test_idx]
            sorted_logits = rank_info.argsort()
            flip_class = sorted_logits[1]
            if flip_class == test_labels[test_idx]:
                flip_class = sorted_logits[0]
        else:
            flip_class = None
        data_support = binary_search(train_idxs, 
                                    flip_class=flip_class,
                                    eval_idx=test_idx,
                                    search_budget=args.search_budget,
                                    num_tests=args.num_tests,
                                    arch=args.arch)
        os.makedirs(args.results_path, exist_ok=True)
        fname = os.path.join(args.results_path, f'{args.test_idx}.pkl')
        with open(fname, 'wb') as f:
            pkl.dump(data_support, f)

if __name__ == '__main__':
    args = parse_args()
    main(args)