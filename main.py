import os
from network.network import make_network
from core.trainer import Trainer

import argparse
import sys
from config.cfg import cfg
import config.cfg as config
from pathlib import Path
from warnings import simplefilter
from data.DBHelperADHD import DBHelper


def pretrain(db_helper, output_directory, max_epoch, k, save_result_per_epoch, beta, delta, gamma, noise_std):
    network = make_network(feature_num=db_helper.norm_total_data.shape[1], k=k, m=0, beta=beta, delta=delta, gamma=gamma, noise_std=noise_std, name='network')

    trainer = Trainer(network=network,
                      db_helper=db_helper,
                      output_directory=output_directory)

    trainer.pretrain_network(max_epoch=max_epoch,
                             save_result_per_epoch=save_result_per_epoch)


def train(args):
    db_helper = DBHelper(args.data)
    db_helper.load_data()

    pass


def test():
    pass


def main():
    simplefilter(action='ignore', category=DeprecationWarning)

    parser = argparse.ArgumentParser(description='eeg representation learning')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Select mode')
    parser.add_argument('--data', type=str, help='data folder directory')
    parser.add_argument('--config', type=str, default='./config/config_file.yaml', help='path to configure file')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help='remainder arguments')
    args = parser.parse_args()

    config.load_cfg(args.config)
    cfg.merge_from_list(args.opts)
    config.assert_cfg()
    cfg.freeze()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test()


if __name__ == '__main__':
    sys.argv = ['main.py',
                '--mode', 'train',
                '--data', '/Users/sangminlee/Documents/YBRAIN/DB', ]
    main()
