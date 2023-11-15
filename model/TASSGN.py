import torch
import torch.nn as nn
import numpy as np
from .TAGEncoder import TAGEncoder
 

class TASSGN(nn.Module):
    """
        TASSGN is complemented based on STID (https://dl.acm.org/doi/pdf/10.1145/3511808.3557702).
    """

    def __init__(self, num_nodes, input_len, output_len, input_dim, hid_dim, num_samples,
            num_layers, num_blocks, num_attention_heads, topk=10, dropout=0.1, time_of_day_size=288, day_of_week_size=7):
        super().__init__()   

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_samples = num_samples
        self.num_layers = num_layers
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

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
        self.sample_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.hid_dim, kernel_size=(1, 1), bias=True)
        self.sample_series_agg_layer = nn.Conv2d(
            in_channels=self.hid_dim * self.num_samples, out_channels=self.hid_dim, kernel_size=(1, 1), bias=True)

        self.tag_encoder = TAGEncoder(input_dim=hid_dim, hidden_dim=hid_dim, output_dim=hid_dim, num_blocks=num_blocks, num_attention_heads=num_attention_heads, num_nodes=num_nodes, topk=topk, dropout=dropout)
        
        self.hidden_dim = hid_dim * 6

        self.encoder = nn.Sequential(*[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
    
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)



    def forward(self, history_data, sample_data):

        """
            history_data: (B, T, N, D) 
            sample_data: (B, S, T, N, D)
        """
        batch_size, history_len, num_nodes, _ = history_data.shape
        num_samples = sample_data.shape[1]


        t_i_d_data = history_data[..., 1]
        d_i_w_data = history_data[..., 2]

        input_data = history_data[..., 0:1]
        sample_data = sample_data[..., 0:1]
        input_data = input_data.reshape(batch_size, history_len, num_nodes, -1)


        time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]

        day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]

        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        time_series_emb = time_series_emb.reshape(batch_size, -1, num_nodes, 1)
        
        # self-sampling data embedding and aggregation as Equation (6)
        sample_data = sample_data.reshape(batch_size*self.num_samples, history_len, num_nodes, 1)
        sample_emb = self.sample_series_emb_layer(sample_data)
        sample_emb = sample_emb.reshape(batch_size, -1, num_nodes, 1)
        sample_emb = self.sample_series_agg_layer(sample_emb)

        # TAG Encoder
        tag_emb = self.tag_encoder(time_series_emb.transpose(1,2).squeeze(-1), sample_emb.transpose(1,2).squeeze(-1))
        tag_emb = tag_emb.transpose(1,2).unsqueeze(-1)

        # attach spatial-temporal identities
        node_emb = []
        node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # concatenate multi-branch embeddings as Equation (21)
        hidden = torch.cat([time_series_emb, sample_emb, tag_emb] + node_emb + tem_emb, dim=1)

        # encoding and forecasting, as shown in Equation (3)(4)
        hidden = self.encoder(hidden)

        prediction = self.regression_layer(hidden)

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

