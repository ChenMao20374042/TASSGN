import numpy as np
import argparse

def generate_time_series_data(data, history_len=12, future_len=12, steps_per_day=288, time_in_day=True, day_in_week=True):
    """
        Generate time series data by a sliding window.
        Args:
            data: in shape (T, N, D), where T is total time steps, N is number of nodes, D is feature dimension.
            history_len: the length of history series (x).
            future_len: the length of future series (y).
            steps_per_day: the number of time steps of a day.
            time_in_day: whether to add time in day features.
            day_in_week: whether to add day in week features.
        Returns:
            x_data: in shape (T', T_h, N, D'), where T' is the number of generated time series, T_h is the length of history series, D' is new feature dimension.
            y_data: in shape (T', T_f, N, D'), where T_f is the length of future series.
            window_data: in shape (T'', T_f, N, D'), where T'' is the number of generated window series.
    """

    total_sequence_len, num_nodes, _ = data.shape
    feature_list = [data]

    # add time in day features
    if time_in_day:
        tid = [i % steps_per_day for i in range(total_sequence_len)]
        tid = np.array(tid)
        tid_tiled = np.tile(tid, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(tid_tiled)

    # add day in week features
    if day_in_week:
        diw = [(i // steps_per_day) % 7 for i in range(total_sequence_len)]
        diw = np.array(diw)
        diw_tiled = np.tile(diw, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(diw_tiled)

    data = np.concatenate(feature_list, axis=-1)

    x_data = []
    y_data = []

    # generate x and y series with sliding window
    for t in range(history_len, total_sequence_len-future_len+1):
        x = data[t-history_len:t]
        y = data[t:t+future_len]
        x_data.append(x)
        y_data.append(y)

    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    # generate series with sliding window for self-sampling
    window_data = []

    for t in range(0, total_sequence_len-future_len+1):
        window_data.append(data[t:t+future_len])

    window_data = np.array(window_data, dtype=float)


    return x_data, y_data, window_data

def generate_train_val_test_data(x_data, y_data, train_ratio=0.6, validation_ratio=0.2):
    """
        Split the time series data into training, validation and test sets.
        Args:
            x_data: history time series data in shape (T', T_h, N, D).
            y_data: future time series data in shape (T', T_f, N, D).
            train_ratio: the ratio of training set.
            validation_ratio: the ratio of validation set.
        Returns:
            a dictionary contains training, validation and test sets, indexed by dictionary['train']['x'] for example.
    """

    test_ratio = 1 - train_ratio - validation_ratio
    
    # check the partition ratio
    assert 0<train_ratio<1 and 0<validation_ratio<1 and 0<test_ratio<1, 'Invalid dataset partition!'

    total_size = x_data.shape[0]
    val_size, test_size = int(total_size * validation_ratio), int(total_size * test_ratio)
    train_size = total_size-val_size-test_size

    # dataset partition
    train_x_data, train_y_data = x_data[0:train_size], y_data[0:train_size]
    val_x_data, val_y_data = x_data[train_size:train_size+val_size], y_data[train_size:train_size+val_size]
    test_x_data, test_y_data = x_data[-test_size:], y_data[-test_size:]

    data = {'train':{'x':train_x_data, 'y':train_y_data},
            'val':{'x':val_x_data, 'y':val_y_data},
            'test':{'x':test_x_data, 'y':test_y_data}}
    
    return data


if __name__ == '__main__':
    
    # arg parse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='PEMS08', help='the name of dataset')
    parser.add_argument('--history_len', type=int, default=12, help='the number of time steps of history series')
    parser.add_argument('--future_len', type=int, default=12, help='the number of time steps of future series')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='the division ratio of training set')
    parser.add_argument('--validation_ratio', type=float, default=0.2, help='the division ratio of validation set')
    parser.add_argument('--time_in_day', type=bool, default=True, help='add time in day features')
    parser.add_argument('--day_in_week', type=bool, default=True, help='add day in week features')
    parser.add_argument('--steps_per_day', type=int, default=288, help='the number of time steps of a day')

    args = parser.parse_args()

    print('start generating time series data from {} with history length {}, future length {}'.format(args.data_name, args.history_len, args.future_len))

    # load PEMS data
    data_dir = './data/{}/'.format(args.data_name)
    data_path = data_dir + args.data_name + '.npz'
    data = np.load(data_path)['data'][..., 0:1] # only traffic flow

    # generate series data
    x_data, y_data, window_data = generate_time_series_data(data, 
                                                history_len=args.history_len,
                                                future_len=args.future_len,
                                                steps_per_day=args.steps_per_day,
                                                time_in_day=args.time_in_day,
                                                day_in_week=args.day_in_week
                                                )
    
    # dataset partition
    data = generate_train_val_test_data(x_data, 
                                        y_data,
                                        train_ratio=args.train_ratio,
                                        validation_ratio=args.validation_ratio)
    
    # save datasets
    for data_type in data.keys():
        for k in data[data_type].keys():
            print('{} {} data shape: {}'.format(data_type, k, data[data_type][k].shape))
            np.save(data_dir+'{}_{}_data.npy'.format(data_type, k), data[data_type][k])

    # save window data
    np.save(data_dir+'window_data.npy', window_data)
    print('window data shape: {}'.format(window_data.shape))

    # save mean and std of training traffic flow, which will be used by standard scaler
    train_x_series = data['train']['x'][..., 0:1]
    ms = np.array([train_x_series.mean(), train_x_series.std()])
    print('mean of train x series: {}, std of train x series: {}'.format(ms[0], ms[1]))
    np.save(data_dir+'mean_std.npy', ms)


