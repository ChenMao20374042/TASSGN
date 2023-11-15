import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import logging
import time
import os

from model.predictor import Predictor
from metrics.metrics import masked_metric, masked_mae, masked_mape, masked_rmse
from utils.utils import load_predictor_datasets, load_scaler, save_predicted_label


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='PEMS08', help='the name of dataset')
    parser.add_argument('--history_len', type=int, default=12, help='the number of time steps of most recent history series')
    parser.add_argument('--input_dim', type=int, default=1, help='input dimension of series')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='the number of encoding layers')
    parser.add_argument('--steps_per_day', type=int, default=288, help='the number of time steps of a day')
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
    logfile = logpath + 'predictor.log'
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
    train_loader, val_loader, scaler, num_clusters = load_predictor_datasets(args.data_name, batch_size=args.batch_size, shuffle=True)
    print('finish loading dataset')

    # device
    device = torch.device(args.device)

    # loss and metrics
    loss_func = nn.CrossEntropyLoss()
    metric = masked_metric

    # init model
    num_nodes = {'PEMS03':358,
                'PEMS04':307,
                'PEMS07':883,
                'PEMS08':170}[args.data_name]

    model = Predictor(num_nodes=num_nodes,
                        input_len=args.history_len,
                        input_dim=args.input_dim,
                        out_dim = num_clusters,
                        hid_dim=args.hid_dim,
                        num_layers=args.num_layers,
                        time_of_day_size=args.steps_per_day,
                        day_of_week_size=7
                        )
    model.to(device)
    
    optimizer = optim.Adam(filter(lambda p : p.requires_grad,model.parameters()),lr=args.learning_rate, weight_decay=args.weight_decay)

    # random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # train and validate
    print('start training predictor...')

    his_accuracy = []
    t1 = time.time()
    
    for epoch in range(1, args.num_epochs+1):

        # train
        model.train()
        train_loss = []
        train_num = []
        train_accurate_num = []

        loop = tqdm(enumerate(train_loader), total =len(train_loader))

        for step, (x, y_real) in loop:

            # predict label
            x = x.to(device)
            y_real = y_real.to(device)
            y_pred = model(x)

            # calculate loss and metrics
            loss = loss_func(y_pred.reshape(-1,num_clusters), y_real.reshape(-1))  
            optimizer.zero_grad() 
            loss.backward()		
            optimizer.step() 
            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            loop.set_postfix(loss=loss.item())
            y_pred = torch.argmax(nn.Softmax(dim=-1)(y_pred), dim=-1).unsqueeze(-1) # figure out predicted label, as shown in Equation (5)
            accurate_num = torch.sum(y_pred == y_real)

            # record metrics
            train_loss.append(loss.item())
            train_accurate_num.append(accurate_num.item())
            train_num.append(y_real.numel())

        # log training info
        info = 'Epoch [{}/{}] Train Loss: {:.4f}, Train Accuracy: {:.4f}'
        info = info.format(epoch, args.num_epochs, np.mean(train_loss), np.sum(train_accurate_num)/np.sum(train_num))
        logger.info(info)

        # validate
        model.eval()
        val_loss = []
        val_num = []
        val_accurate_num = []

        loop = tqdm(enumerate(val_loader), total =len(val_loader))
        for step, (x, y_real) in loop:

            # predict label
            x = x.to(device)
            y_real = y_real.to(device)
            with torch.no_grad():
                y_pred = model(x)

            # calculate loss and metrics
            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            loss = loss_func(y_pred.reshape(-1,num_clusters), y_real.reshape(-1)) 
            loop.set_postfix(loss=loss.item())
            y_pred = torch.argmax(nn.Softmax(dim=-1)(y_pred), dim=-1).unsqueeze(-1) # figure out predicted label, as shown in Equation (5)
            accurate_num = torch.sum(y_pred == y_real)

            # record metrics
            val_loss.append(loss.item())
            val_accurate_num.append(accurate_num.item())
            val_num.append(y_real.numel())

        # log validation info
        info = 'Epoch [{}/{}] Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'
        info = info.format(epoch, args.num_epochs, np.mean(val_loss), np.sum(val_accurate_num)/np.sum(val_num))
        logger.info(info)

        # save model
        if epoch == 1 or np.sum(val_accurate_num)/np.sum(val_num) > np.max(his_accuracy):
            torch.save(model.state_dict(), save_path+'predictor_best_val_accuracy.pth')
            logger.info('best validation model has been saved.')
        
        his_accuracy.append(np.sum(val_accurate_num)/np.sum(val_num))

    # finish training
    t2 = time.time()
    logger.info('finish training, total training time: {}'.format(t2-t1))

    # save predicted label on cpu (in case of CUDA OUT OF MEMORY)
    print('start saving predicted labels')

    device = torch.device('cpu') 
    best_val_params = torch.load(save_path+'predictor_best_val_accuracy.pth')
    model.load_state_dict(best_val_params)
    model.to(device)

    pred_labels = save_predicted_label(args.data_name, model)

    logger.info('finish saving predicted labels.')
    logger.info('predicted labels shape: {}'.format(pred_labels.shape))
