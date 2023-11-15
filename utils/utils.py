import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

class StandardScaler():
    """
    Standardize the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class TimeSeriesDataset(Dataset):
    def __init__(self, x_data, y_data, window_data=None, sample_index=None):

        super().__init__()
        assert x_data.shape[0] == y_data.shape[0], 'x data and y data should have same size.'
       
        size, self.history_len, self.num_nodes, self.num_features = x_data.shape
        self.future_len = y_data.shape[1]
        

        self.x_data = torch.tensor(x_data, dtype=torch.float)
        self.y_data = torch.tensor(y_data, dtype=torch.float)

        # self-sampling
        if sample_index is not None:
            self.num_samples = sample_index.shape[2]
            self.sample_index = torch.tensor(sample_index, dtype=torch.long)
            self.window_data = torch.tensor(window_data, dtype=torch.float)
            self.window_data = self.window_data.transpose(1,2).reshape(-1, self.future_len, self.num_features) # (B*N, L, D)
            if sample_index.shape[0] < self.x_data.shape[0]:
                self.x_data = self.x_data[-sample_index.shape[0]:]
                self.y_data = self.y_data[-sample_index.shape[0]:]
        else:
            self.sample_index = None

    def __getitem__(self, index) -> tuple:
        x = self.x_data[index]
        y = self.y_data[index]
        
        # self-sampling
        if self.sample_index is not None:
            sample_index = self.sample_index[index].reshape(-1)
            sample_data = self.window_data[sample_index, ...]
            sample_data = sample_data.reshape(self.num_nodes, self.num_samples, self.future_len, self.num_features).permute([1,2,0,3])
            return x, sample_data, y

        else:
            return x, y

    
    def __len__(self):
        return len(self.x_data)


# The following functions are used to load datasets for different components and save some intermediate results.
# ATTENTION: the traffic flow in x datasets is standardized.

def load_scaler(data_name):

    mean_std = np.load('./data/{}/mean_std.npy'.format(data_name))
    mean, std = mean_std[0], mean_std[1]
    scaler = StandardScaler(mean, std)

    return scaler

def load_encoder_datasets(data_name, batch_size, shuffle=True):

    data_dir = './data/{}/'.format(data_name)

    scaler = load_scaler(data_name)

    train_y_data = np.load(data_dir+'train_y_data.npy')
    train_x_data = train_y_data.copy() # deep clone
    train_x_data[..., 0] = scaler.transform(train_x_data[..., 0]) # standardize

    val_y_data = np.load(data_dir+'val_y_data.npy')
    val_x_data = val_y_data.copy() # deep clone
    val_x_data[..., 0] = scaler.transform(val_x_data[..., 0]) # standardize

    train_dataset = TimeSeriesDataset(x_data=train_x_data, y_data=train_y_data[...,0:1])
    val_dataset = TimeSeriesDataset(x_data=val_x_data, y_data=val_y_data[...,0:1])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle)

    return train_dataloader, val_dataloader, scaler

def save_encoder_representation(data_name, encoder):

    data_dir = './data/{}/'.format(data_name)

    # load y data
    train_y_data = np.load(data_dir+'train_y_data.npy')
    val_y_data = np.load(data_dir+'val_y_data.npy')
    train_y_data = torch.tensor(train_y_data, dtype=torch.float)
    val_y_data = torch.tensor(val_y_data, dtype=torch.float)

    # standardize
    scaler = load_scaler(data_name)
    train_y_data[..., 0] = scaler.transform(train_y_data[..., 0]) # standardize
    val_y_data[..., 0] = scaler.transform(val_y_data[..., 0]) # standardize

    # encode
    with torch.no_grad():
        train_encode = encoder.encode(train_y_data)
        val_encode = encoder.encode(val_y_data)

    train_encode = train_encode.numpy()
    val_encode = val_encode.numpy()

    np.save(data_dir+'train_representation.npy', train_encode)
    np.save(data_dir+'val_representation.npy', val_encode)

    return train_encode, val_encode

def load_labeler_datasets(data_name):

    data_dir = './data/{}/'.format(data_name)

    train_representation = np.load(data_dir+'train_representation.npy')
    val_representation = np.load(data_dir+'val_representation.npy')

    return train_representation, val_representation


def load_predictor_datasets(data_name, batch_size, shuffle=True):

    data_dir = './data/{}/'.format(data_name)

    scaler = load_scaler(data_name)

    train_x_data = np.load(data_dir+'train_x_data.npy')
    train_x_data[..., 0] = scaler.transform(train_x_data[..., 0]) # standardize
    train_y_data = np.load(data_dir+'train_cluster_label.npy')

    val_x_data = np.load(data_dir+'val_x_data.npy')
    val_x_data[..., 0] = scaler.transform(val_x_data[..., 0]) # standardize
    val_y_data = np.load(data_dir+'val_cluster_label.npy')

    # the maximum number of clusters, which is the output dimension of predictor
    num_clusters = int(np.max(train_y_data)) + 1 

    train_dataset = TensorDataset(torch.tensor(train_x_data, dtype=torch.float), torch.tensor(train_y_data, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_x_data, dtype=torch.float), torch.tensor(val_y_data, dtype=torch.long))

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle)

    return train_dataloader, val_dataloader, scaler, num_clusters

def save_predicted_label(data_name, predictor):

    data_dir = './data/{}/'.format(data_name)

    data_types = ['train', 'val', 'test']
    pred_labels = []

    scaler = load_scaler(data_name)

    for data_type in data_types:
        # load x data
        x_data = np.load(data_dir+'{}_x_data.npy'.format(data_type))
        x_data = torch.tensor(x_data, dtype=torch.float)

        # standardize
        x_data[..., 0] = scaler.transform(x_data[..., 0])

        # predict
        with torch.no_grad():
            y_pred = predictor(x_data)
        y_pred = torch.argmax(nn.Softmax(dim=-1)(y_pred), dim=-1).unsqueeze(-1)
        pred_labels.append(y_pred.numpy())
    
    pred_labels = np.concatenate(pred_labels, axis=0)

    np.save(data_dir+'predicted_label.npy', pred_labels)

    return pred_labels


def load_forecasting_datasets(data_name, batch_size, shuffle=True):

    data_dir = './data/{}/'.format(data_name)

    data_types = ['train', 'val', 'test']
    dataloaders = []

    scaler = load_scaler(data_name)

    window_data = np.load(data_dir + 'window_data.npy')
    window_data[..., 0] = scaler.transform(window_data[..., 0]) # standardize

    for data_type in data_types:
        x_data = np.load(data_dir+'{}_x_data.npy'.format(data_type))
        x_data[..., 0] = scaler.transform(x_data[..., 0]) # standardize
        y_data = np.load(data_dir+'{}_y_data.npy'.format(data_type))
        sample_index = np.load(data_dir+'{}_sample_index.npy'.format(data_type))
        dataset = TimeSeriesDataset(x_data=x_data, y_data=y_data[..., 0:1], window_data=window_data, sample_index=sample_index)
        dataloader = DataLoader(dataset, batch_size, shuffle)
        dataloaders.append(dataloader)
    
    return dataloaders, scaler

