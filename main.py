import argparse

import os
import sys
from train import *



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument('--batch_size', '-bs', default = 32, type = int,
                        help = 'mini batches size')


    parser.add_argument('--epoch', '-e', default = 100, type = int,
                        help = 'num of epochs')


    parser.add_argument('--learning_rate', '-lr', default = 1e-3, type = float,
                        help = 'learning rate')


    parser.add_argument('--data_path', '-dp',
                        default = "../mapillary", type = str,
                        help = 'path to mapillary dataset. It should be like PATH/dataset')

    parser.add_argument('--run_name', '-rn',
                        default = "mapillary", type = str,
                        help = 'the run name will be apeared in wandb')

    parser.add_argument('--project_name', '-pn',
                        default = "unet", type = str,
                        help = 'the run name will be apeared in wandb')

    parser.add_argument('--continue_tra', '-ct',
                        default = False, type = bool,
                        help = 'train a model for more epochs. you also need to set the model path.')

    parser.add_argument('--model_path', '-mp',
                        default = "./checkpoints/test/best.pth", type = str,
                        help = 'The path to the model is going to be trained.')

    parser.add_argument('--wandb_id', '-id',
                        default = "test", type = str,
                        help = 'Corresponding wandb run id to resume training.')


    args = parser.parse_args()

    train(args.batch_size, args.epoch, args.learning_rate, args.run_name, args.data_path, args.project_name, args.continue_tra, args.model_path, args.wandb_id)



if __name__ == "__main__":
    main()
