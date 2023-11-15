import numpy as np
import torch
import argparse

def generate_self_sampling_index(pred_label, num_samples=7, history_len=12, future_len=12):
    """ Generate the self-sampling series based on predicted labels. 
        To reduce storage occupation, only store the index of sampled series in window series data.
        If history series is insufficient to sample, the series will be padded with recent history time steps.

    Args:
        pred_label: the predicted labels in shape (B, N, 1) where B is the size of data, N is the number of nodes.
        num_samples: the number of history series that will be sampled.
        history_len: the number of history time steps.
        future_len: the number of future time steps.

    Returns:
        the index of sampled series, in shape (B, N, S, 1), where S equals to num_samples.
    """

    total_size, num_nodes, _ = pred_label.shape
    pred_label = torch.tensor(pred_label, dtype=torch.int)
    print('predicted label shape: {}'.format(pred_label.shape))

    max_label = torch.max(pred_label)+1

    clusters = [[[] for i in range(max_label)] for j in range(num_nodes)]

    sample_index = [] # the index of sampled series

    for t in range(0, total_size):

        # no history series to sample at the begining
        if t + history_len - future_len < 0:
            continue
        
        node_sample_index = [[] for i in range(num_nodes)]
        for node_id in range(num_nodes):
            label = pred_label[t, node_id].item() # predicted label

            #                    Explanation about the index 
            #
            # Time series: | | | | | | | | | | | | | | | | | | | | | | | | |
            #              |<-------- x ---------->|<-------- y ---------->|
            #              ^                       ^
            #              |                       |
            # Time index:  t                    t + history_len
            #
            # The window data in shape (total_size, num_nodes, future_len, num_features)
            # When forecasting, we firstly reshape window data as (total_time_steps*num_nodes, future_len, num_features).
            # So the two-dimensional index (t, node_id) should be transformed into one-dimensional index (t*num_nodes + node_id).

            # Sample S recent history y series that share the same predicted label.
            # Make sure that all sampled history series appear earlier than y series to AVOID INFORMATION LEAKAGE.
            sample_history_y = [idx for idx in clusters[node_id][label] if idx <= (t + history_len - future_len) * num_nodes + node_id] 

            # if no history series that share the same label, then replace with most recent history series
            if len(sample_history_y) == 0:
                sample_history_y.append((t + history_len - future_len) * num_nodes + node_id)
            
            # repeat the sampled indexes to expand its size
            while len(sample_history_y) < num_samples:
                sample_history_y += sample_history_y

            node_sample_index[node_id] += sample_history_y[-num_samples:]

            # add y index into corresponding cluster
            clusters[node_id][label].append((t + history_len) * num_nodes + node_id)
        
        sample_index.append(node_sample_index)
        print('generating self-sampling index: {}/{}'.format(t+1, total_size), end = '\r')
    
    sample_index = np.array(sample_index)
    sample_index = sample_index.reshape(-1, num_nodes, num_samples, 1)
    print('self-sampling index shape: {}'.format(sample_index.shape))
    return sample_index

def generate_train_val_test_index(sample_index, total_size, train_ratio=0.6, validation_ratio=0.2):
    """ Split the sampled index into training, validation and test sets.

    Args:
        sample_index: the index of self-sampling series, in shape (B, N, S, 1).
        total_size: the size of time series data (the sum of the size of training, validation and test series)
                    the size of sample_index may be smaller than total_size when future_len > history_len, 
                    as the self-sampling indexes cannot be generated for the begining series.
        train_ratio: the ratio of training dataset.
        validation_ratio: the ratio of validation dataset.

    Returns:
        a dictionary containing the training, validation and test sets.
    """

    test_ratio = 1 - train_ratio - validation_ratio
    
    # check the partition ratio
    assert 0<train_ratio<1 and 0<validation_ratio<1 and 0<test_ratio<1, 'Invalid dataset partition!'

    # size of sample_index may be smaller than total_size if future_len > history_len.
    # in this case, the begining of training dataset would be truncated, as their sample index cannot be generated.
    val_size, test_size = int(total_size * validation_ratio), int(total_size * test_ratio)
    train_size = sample_index.shape[0] - val_size - test_size

    # dataset partition
    train_sample_index = sample_index[:train_size]
    val_sample_index = sample_index[train_size:train_size+val_size]
    test_sample_index = sample_index[-test_size:]

    print('train sample index: {}'.format(train_sample_index.shape))
    print('validation sample index: {}'.format(val_sample_index.shape))
    print('test sample index: {}'.format(test_sample_index.shape))

    index = {'train':train_sample_index,
            'val':val_sample_index,
            'test':test_sample_index}
    
    return index


if __name__ == '__main__':
    
    # arg parse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='PEMS08', help='the name of dataset')
    parser.add_argument('--history_len', type=int, default=12, help='the number of time steps of history series')
    parser.add_argument('--future_len', type=int, default=12, help='the number of time steps of future series')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='the division ratio of training set')
    parser.add_argument('--validation_ratio', type=float, default=0.2, help='the division ratio of validation set')
    parser.add_argument('--num_samples', type=int, default=7, help='the number of self-sampling history series')

    args = parser.parse_args()     

    print('start generating self-sampling index from {} with history length {}, future length {}, number of samples {}'.format(args.data_name, args.history_len, args.future_len, args.num_samples))       

    # load predicted label
    data_dir = './data/{}/'.format(args.data_name)
    pred_label = np.load(data_dir+'predicted_label.npy')

    # self-sampling
    sample_index = generate_self_sampling_index(pred_label, num_samples=args.num_samples, history_len=args.history_len, future_len=args.future_len)

    # dataset partition
    index = generate_train_val_test_index(sample_index, total_size=pred_label.shape[0], train_ratio=args.train_ratio, validation_ratio=args.validation_ratio)

    # save
    for data_type in index.keys():
        np.save(data_dir+'{}_sample_index.npy'.format(data_type), index[data_type])












