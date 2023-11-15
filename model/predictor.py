from .encoder import STIDEncoder
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class Predictor(nn.Module):
    """ Predictor in Self-Sampling, which takes history series x as input to predict the label of future series y.
    """

    def __init__(self, num_nodes, input_len, input_dim, hid_dim, out_dim,
            num_layers, time_of_day_size=288, day_of_week_size=7):
        super().__init__()
        
        """
        Args:
            num_nodes: the number of nodes.
            input_len: the length of input series (history series x).
            input_dim: the input dimension (not including time_in_day and day_in_week features)
            hid_dim: hidden dimension.
            out_dim: output dimension, which equals to the number of different labels.
            num_layers: the number of encoding layers in STID encoder.
            time_of_day_size: the number of time steps of a day.
            day_of_week_size: the number of days of a week.
        """

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers  
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.out_dim = out_dim

        self.encoder = STIDEncoder(num_nodes=num_nodes,
                                    input_len=input_len, 
                                    output_len=out_dim,
                                    input_dim=input_dim, 
                                    hid_dim=hid_dim, 
                                    num_layers=num_layers, 
                                    time_of_day_size=time_of_day_size, 
                                    day_of_week_size=day_of_week_size)
        
        # node-specific MLP layers
        self.w1 = nn.Parameter(torch.rand([num_nodes, self.hid_dim*4, self.hid_dim*4]), requires_grad=True)
        self.b1 = nn.Parameter(torch.rand([num_nodes, self.hid_dim*4]), requires_grad=True)

        self.w2 = nn.Parameter(torch.rand([num_nodes, self.hid_dim*4, self.out_dim]), requires_grad=True)
        self.b2 = nn.Parameter(torch.rand([num_nodes, self.out_dim]), requires_grad=True)

        self.tanh = nn.Tanh()

    def forward(self, history_data):
        """ Predict the label of future series from history series.

        Args:
            history_data: in shape (B, T, N, 3) where B is batch size, T is the number of time steps of history series, N is the number of nodes.
        
        Returns:
            predicted label distribution, in shape (B, N, C), where C is the number of different labels.
        """
        
        B, T, N, C = history_data.shape
        
        encode_data = self.encoder.encode(history_data, mask=False)

        encode_data = encode_data.reshape(B, -1, N).transpose(1,2)
        

        encode_data = encode_data.transpose(0,1)

        # node-specific MLP as shown in Equation (5) except for softmax and argmax.
        encode_data = torch.matmul(encode_data, self.w1) + self.b1.unsqueeze(1)
        encode_data = self.tanh(encode_data)
        encode_data = torch.matmul(encode_data, self.w2) + self.b2.unsqueeze(1)
        encode_data = encode_data.transpose(0,1)

        return encode_data

