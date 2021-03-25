import argparse

import os
import sys
from train import *



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument('--batch_size', '-bs', default = 15, type = int,
                        help = 'mini batches size')


    parser.add_argument('--epoch', '-e', default = 100, type = int,
                        help = 'num of epochs')


    parser.add_argument('--learning_rate', '-lr', default = 1e-3, type = float,
                        help = 'learning rate')


    parser.add_argument('--data_path', '-dp',
                        default = "/home/lunet/wsmo6/mapillary/dataset", type = str,
                        help = 'path to mapillary dataset. It should be like PATH/dataset')

    parser.add_argument('--run_name', '-rn',
                        default = "mapillary", type = str,
                        help = 'the run name will be apeared in wandb')


    args = parser.parse_args()

    train(args.batch_size, args.epoch, args.learning_rate, args.run_name,
            args.data_path)



if __name__ == "__main__":
    main()
