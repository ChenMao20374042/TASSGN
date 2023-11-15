import torch
import torch.nn as nn
import random

class STIDEncoder(nn.Module):
    """ STID Encoder in Self-Sampling, used to generate spatial-temporal representation.
    """
    def __init__(self, num_nodes, input_len, output_len, input_dim, hid_dim, 
            num_layers, time_of_day_size=288, day_of_week_size=7, mask_ratio=0.5):
        super().__init__()   
        """
        Args:
            num_nodes: the number of nodes.
            input_len: the length of input series (future series y).
            input_dim: the input dimension (not including time_in_day and day_in_week features)
            hid_dim: hidden dimension.
            out_dim: output dimension, which equals to input_len.
            num_layers: the number of encoding layers.
            time_of_day_size: the number of time steps of a day.
            day_of_week_size: the number of days of a week.
            mask_ratio: the mask ratio in self-supervised training.
        """
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.mask_ratio = mask_ratio   
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.mask_ratio = mask_ratio

        self.node_emb = nn.Parameter(
            torch.empty(self.num_nodes, self.hid_dim))
        nn.init.xavier_uniform_(self.node_emb)

        self.time_in_day_emb = nn.Parameter(
            torch.empty(self.time_of_day_size, self.hid_dim))
        nn.init.xavier_uniform_(self.time_in_day_emb)

        self.day_in_week_emb = nn.Parameter(
            torch.empty(self.day_of_week_size, self.hid_dim))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.hid_dim, kernel_size=(1, 1), bias=True)

        self.hidden_dim = hid_dim * 4

        self.encoder = nn.Sequential(*[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def mask(self, input_data):
        """ Mask the input data for self-supervised training.

        Args:
            input_data: in shape (B, T, N, 3) where B is batch size, T is the number of time steps, N is the number of nodes.

        Returns:
            None, as directly mask the original input tensor.
        """
        mask_num = int(self.input_len * self.mask_ratio)
        mask_index = random.sample(range(self.input_len), mask_num)
        input_data[:, :, mask_index, :] = 0

    def encode(self, history_data, mask=False):
        """ Encode the series data and generate spatial-temporal representation as descibed in Equation (1) to (3).
            
        Args:
            history_data: in shape (B, T, N, 3)
            mask: if mask=True, then the input series will be randomly masked for self-supervised training.

        Returns:
            the spatial-temporal representation in shape (B, 4D, N, 1) where D is the hidden dimension.
        """

        input_data = history_data[..., 0:1] # traffic flow

        if mask:
            self.mask(input_data)
        t_i_d_data = history_data[..., 1] # time in day feature

        time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]

        d_i_w_data = history_data[..., 2] # day in week feature
        day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data) # Equation (1)

        node_emb = [] # node embedding
        node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # attach spatial-temporal identities
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1) # Equation (2)

        # residual MLP encoding layers
        hidden = self.encoder(hidden) # Equation (3)

        return hidden

    def pretrain(self, histor_data):
        """Self-supervised training, as the input series is randomly masked and the encoder is pretrained to reconstruct the series.
            
        Args:
            history_data: in shape (B, T, N, 3)

        Returns:
            reconstructed series in shape (B, T, N, 1). only reconstructe the traffic flow, no time_in_day and day_in_week features.
        """

        hidden = self.encode(histor_data, mask=True)

        prediction = self.regression_layer(hidden) # Equation (4)

        return prediction



class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

