import numpy as np
from tqdm import tqdm
import argparse
import logging
import time
import os

from model.labeler import Labeler
from utils.utils import load_labeler_datasets
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='PEMS08', help='the name of dataset')
    parser.add_argument('--experiment_id', type=int, default=0, help='the id of current experiment, used to save log and model')
    parser.add_argument('--thresh', type=int, default=30, help='the threshold of binary cluster')
    parser.add_argument('--repeat', type=int, default=10, help='the number of repetitions of clustering algorithm training.')

    args = parser.parse_args()

    # logging
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    logpath = './logs/exp_{}/'.format(args.experiment_id)
    logfile = logpath + 'labeler.log'
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    handler = logging.FileHandler(logfile, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    # model saving path
    save_path = './garage/exp_{}/'.format(args.experiment_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # load dataset
    train_data, val_data = load_labeler_datasets(args.data_name)
    print('finish loading dataset')

    # init model
    num_nodes = {'PEMS03':358,
                'PEMS04':307,
                'PEMS07':883,
                'PEMS08':170}[args.data_name]

    model = Labeler(num_nodes=num_nodes, thresh=args.thresh, repeat=args.repeat)

    # generate label
    print('start generating labels...')
    t1 = time.time()

    train_num, val_num = train_data.shape[0], val_data.shape[0]
    train_labels = np.empty([train_num, 0, 1])
    train_index = np.array(range(train_num))
    val_labels = np.empty([val_num, 0, 1])
    val_index = np.array(range(val_num))

    for i in range(num_nodes):
        train_x = train_data[:, :, i, 0]
        val_x = val_data[:, :, i, 0]

        train_node_labels = np.zeros([train_num,1], dtype=np.int32)
        val_node_labels = np.zeros([val_num,1], dtype=np.int32)
   
        model.label_counter = 0 # clear label counter
        model.binary_cluster(train_x, train_index, train_node_labels, val_x, val_index, val_node_labels) # apply binary cluster on node i

        # concat the labels of node i
        train_labels = np.concatenate([train_labels, train_node_labels.reshape(train_num,1,1)], axis=1)   
        val_labels = np.concatenate([val_labels, val_node_labels.reshape(val_num,1,1)], axis=1) 

        logger.info('finish cluster node {}/{}. cluster number: {}.'.format(i+1, num_nodes, model.label_counter))
    

    t2 = time.time()
    logger.info('finish generating labels, total spent time: {}'.format(t2-t1))

    # save generated label
    data_dir = './data/{}/'.format(args.data_name)
    np.save(data_dir+'train_cluster_label.npy', train_labels)
    np.save(data_dir+'val_cluster_label.npy', val_labels)
    

