import os
import time
# ignoring TF info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

import argparse

from data_loader import load_dataset
from train import train
from test import test

DATASETS = ['ids17', 'ids18']

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f} ---".format(int(hours), int(minutes), seconds))


def run_experiment(args):
    """
    Running a specific experiment with the given args
    Parameters
    ----------
    args: argparse args. The args given to the program
    """

    # setting seed
    tf.random.set_seed(1234)
    np.random.seed(1234)

    # loading the preprocessed dataset
    print(f"--- Loading preprocessed {args.dataset_name} dataset ---")
    start_time = time.time()
    train_ds, test_ds, features_dim = load_dataset(args)
    end_time = time.time()
    print(f"--- {args.dataset_name} dataset ready after: ", end='')
    get_execute_time(start_time, end_time)

    # training
    print("--- Start training ---")
    start_time = time.time()
    trained_estimator = train(train_ds, features_dim, args)
    end_time = time.time()
    print("--- Training finished after: ", end='')
    get_execute_time(start_time, end_time)

    # testing
    print("--- Start testing ---")
    start_time = time.time()
    test(test_ds, trained_estimator, args)
    end_time = time.time()
    print("--- Testing finished after: ", end='')
    get_execute_time(start_time, end_time)

if __name__ == '__main__':

    # getting the dataset to preprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, dest='dataset_name', type=str, help='the dataset to preprocess and save to disk for later use')
    parser.add_argument('-w', "--windowsize", dest='window_size', type=int, help="The size of the sliding window to know which preprocessed file to load")
    parser.add_argument("-e", "--epochs", dest="num_epochs", default=300, type=int, help="The number of epochs to train with the anomaly detector model")
    parser.add_argument('-b', "--batchsize", dest="batch_size", default=32, type=int, help="The batch size used for training")
    args = parser.parse_args()

    # adding support for runnning all of the available datasets
    if args.dataset_name == "all":
        dataset_index = int(os.environ['SLURM_ARRAY_TASK_ID'])-1
        args.dataset_name = DATASETS[dataset_index]
    
    # running experiments with the given program arguments
    run_experiment(args)