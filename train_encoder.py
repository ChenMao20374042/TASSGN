import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import logging
import time
import os

from model.encoder import STIDEncoder
from metrics.metrics import masked_metric, masked_mae, masked_mape, masked_rmse
from utils.utils import load_encoder_datasets, save_encoder_representation, load_scaler


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='PEMS08', help='the name of dataset')
    parser.add_argument('--future_len', type=int, default=12, help='the number of time steps of future series')
    parser.add_argument('--input_dim', type=int, default=1, help='input dimension of series')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='the number of encoding layers')
    parser.add_argument('--steps_per_day', type=int, default=288, help='the number of time steps of a day')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio of self-supervised training')
    parser.add_argument('--experiment_id', type=int, default=0, help='the id of current experiment, used to save log and model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='the number of training epoches')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()

    # logging
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    logpath = './logs/exp_{}/'.format(args.experiment_id)
    logfile = logpath + 'encoder.log'
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
    train_loader, val_loader, scaler = load_encoder_datasets(args.data_name, batch_size=args.batch_size, shuffle=True)
    print('finish loading dataset')

    # device
    device = torch.device(args.device)

    # loss and metrics
    loss_func = masked_mae
    metric = masked_metric

    # init model
    num_nodes = {'PEMS03':358,
                'PEMS04':307,
                'PEMS07':883,
                'PEMS08':170}[args.data_name]

    model = STIDEncoder(num_nodes=num_nodes,
                        input_len=args.future_len,
                        output_len=args.future_len,
                        input_dim=args.input_dim,
                        hid_dim=args.hid_dim,
                        num_layers=args.num_layers,
                        time_of_day_size=args.steps_per_day,
                        day_of_week_size=7,
                        mask_ratio=args.mask_ratio
                        )
    model.to(device)
    
    optimizer = optim.Adam(filter(lambda p : p.requires_grad,model.parameters()),lr=args.learning_rate, weight_decay=args.weight_decay)

    # random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # train and validate
    print('start training...')

    his_loss = []
    t1 = time.time()
    
    for epoch in range(1, args.num_epochs+1):

        # train
        model.train()
        train_loss = []
        train_mape = []
        train_rmse = []

        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for step, (x, y_real) in loop:

            # reconstruct
            x = x.to(device)
            y_real = y_real.to(device)
            y_pred = model.pretrain(x)
            y_pred = scaler.inverse_transform(y_pred) # inverse transform

            # calculate loss and metrics
            loss = loss_func(y_pred, y_real, null_val=0.0) 
            optimizer.zero_grad() 
            loss.backward()		
            optimizer.step() 
            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            mae, mape, rmse = metric(y_pred, y_real, null_val=0)
            loop.set_postfix(mae=mae, mape=mape, rmse=rmse)

            # record metrics
            train_loss.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)

        # log training info
        info = 'Epoch [{}/{}] Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
        info = info.format(epoch, args.num_epochs, np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse))
        logger.info(info)

        # validate
        model.eval()
        val_loss = []
        val_mape = []
        val_rmse = []

        loop = tqdm(enumerate(val_loader), total =len(val_loader))
        for step, (x, y_real) in loop:

            # reconstruct
            x = x.to(device)
            y_real = y_real.to(device)
            with torch.no_grad():
                y_pred = model.pretrain(x)
            y_pred = scaler.inverse_transform(y_pred) # inverse transform

            # calculate metrics
            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            mae, mape, rmse = metric(y_pred, y_real, null_val=0)
            loop.set_postfix(mae=mae, mape=mape, rmse=rmse)

            # record metrics
            val_loss.append(mae)
            val_mape.append(mape)
            val_rmse.append(rmse)
        
        # log validation info
        info = 'Epoch [{}/{}] Validation Loss: {:.4f}, Validation MAPE: {:.4f}, Validation RMSE: {:.4f}'
        info = info.format(epoch, args.num_epochs, np.mean(val_loss), np.mean(val_mape), np.mean(val_rmse))
        logger.info(info)

        # save model
        if epoch == 1 or np.mean(val_loss) < np.min(his_loss):
            torch.save(model.state_dict(), save_path+'encoder_best_val_loss.pth')
            logger.info('best validation model has been saved.')
        
        his_loss.append(np.mean(val_loss))

    # finish training
    t2 = time.time()
    logger.info('finish training, total training time: {}'.format(t2-t1))

    # save representation on cpu (in case of CUDA OUT OF MEMORY)
    logger.info('start saving encoded representation')

    device = torch.device('cpu')
    best_val_params = torch.load(save_path+'encoder_best_val_loss.pth')
    model.load_state_dict(best_val_params)
    model.to(device)

    train_encode, val_encode = save_encoder_representation(args.data_name, model)

    logger.info('finish saving encoded representation')
    logger.info('training representaion shape: {}'.format(train_encode.shape))
    logger.info('validation representaion shape: {}'.format(val_encode.shape))

