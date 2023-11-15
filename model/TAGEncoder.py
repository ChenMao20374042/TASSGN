import torch
from torch import nn
import numpy as np

class GraphConvolutionLayer(nn.Module):

    def __init__(self, hidden_dim, num_nodes, topk, nodevec1=None, nodevec2=None, dropout=0.1):
        super(GraphConvolutionLayer, self).__init__()

        self.topk = topk

        self.nodevec1 = nn.Parameter(torch.rand([num_nodes, hidden_dim]), requires_grad=True) if nodevec1 is None else nodevec1
        self.nodevec2 = nn.Parameter(torch.rand([num_nodes, hidden_dim]), requires_grad=True) if nodevec2 is None else nodevec2

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def sparse_graph_convolution(self, src):

        # sparse graph convolution as shown in Equation (7)
        adj = torch.matmul(self.nodevec1, self.nodevec2.T)

        # sparsify adjacent matrix
        max_indices = torch.topk(adj, k=self.topk, dim=-1).indices
        sparse_adj = torch.ones_like(adj).to(adj.device) * -float('inf')
        sparse_adj.scatter_(dim=-1, index=max_indices, src=adj.gather(dim=-1, index=max_indices))
        sparse_adj = self.softmax(sparse_adj)

        output = torch.matmul(sparse_adj, src)

        return output
    
    def feed_forward(self, x):
        x = self.linear2(self.relu(self.dropout(self.linear1(x))))
        return self.dropout(x)
    
    def forward(self, src, tgt):

        tgt = self.norm1(tgt + self.sparse_graph_convolution(src)) # Equation (8)
        tgt = self.norm2(tgt + self.feed_forward(tgt)) # Equation (9)

        return tgt

class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_attention_heads, topk, num_nodes, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.topk = topk
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def sparse_attention(self, src, tgt):
        
        batch_size = src.shape[0]

        # multi-head attention, as shown in Equation (10) (11)
        query = self.query_layer(tgt)
        key = self.key_layer(src)
        value = self.value_layer(src)
        query = torch.cat(torch.split(query, self.num_attention_heads, dim=-1), dim=0) / (self.hidden_dim**0.5)
        key = torch.cat(torch.split(key, self.num_attention_heads, dim=-1), dim=0).transpose(1,2)
        value = torch.cat(torch.split(value, self.num_attention_heads, dim=-1), dim=0)
        attn_score = torch.matmul(query, key)

        # sparsify attention scores
        max_indices = torch.topk(attn_score, k=self.topk, dim=-1).indices
        sparse_attn_score = torch.ones_like(attn_score).to(attn_score.device) * -float('inf')
        sparse_attn_score.scatter_(dim=-1, index=max_indices, src=attn_score.gather(dim=-1, index=max_indices))
        sparse_attn_score = self.softmax(sparse_attn_score)

        attn_output = torch.matmul(sparse_attn_score, value)
        attn_output = torch.cat(torch.split(attn_output, batch_size, dim=0), dim=-1)

        return attn_output
    
    def feed_forward(self, x):
        x = self.linear2(self.relu(self.dropout(self.linear1(x))))
        return self.dropout(x)

    def forward(self, src, tgt):

        tgt = self.norm1(tgt + self.sparse_attention(src, tgt)) # Equation (12)
        tgt = self.norm2(tgt + self.feed_forward(tgt)) # Equation (13)

        return tgt


class SSGBlock(nn.Module):
    def __init__(self, hidden_dim, num_nodes, num_attention_heads, topk=10, nodevec1=None, nodevec2=None, dropout=0.1):
        super(SSGBlock, self).__init__()
        self.conv = GraphConvolutionLayer(hidden_dim=hidden_dim, num_nodes=num_nodes, topk=topk, nodevec1=nodevec1, nodevec2=nodevec2, dropout=dropout)
        self.attn = GraphAttentionLayer(hidden_dim=hidden_dim, num_attention_heads=num_attention_heads, num_nodes=num_nodes, topk=topk, dropout=dropout)
    
    def forward(self, src, tgt):

        tgt = self.conv(src=src, tgt=tgt)

        tgt = self.attn(src=src, tgt=tgt)

        return tgt

class TAGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks, num_nodes, num_attention_heads, topk=10, dropout=0.1):
        super(TAGEncoder, self).__init__()

        self.src_embedding = nn.Linear(input_dim, hidden_dim)
        self.tgt_embedding = nn.Linear(input_dim, hidden_dim)

        # shared structure parameters
        self.src_nodevec1 = nn.Parameter(torch.rand([num_nodes, hidden_dim]), requires_grad=True)
        self.src_nodevec1 = nn.init.xavier_uniform_(self.src_nodevec1)
        self.src_nodevec2 = nn.Parameter(torch.rand([num_nodes, hidden_dim]), requires_grad=True)
        self.src_nodevec2 = nn.init.xavier_uniform_(self.src_nodevec2)
        self.tgt_nodevec1 = nn.Parameter(torch.rand([num_nodes, hidden_dim]), requires_grad=True)
        self.tgt_nodevec1 = nn.init.xavier_uniform_(self.tgt_nodevec1)
        self.tgt_nodevec2 = nn.Parameter(torch.rand([num_nodes, hidden_dim]), requires_grad=True)
        self.tgt_nodevec2 = nn.init.xavier_uniform_(self.tgt_nodevec2)

        self.history_blocks = nn.ModuleList() # the history graph
        self.transition_blocks = nn.ModuleList() # the transition graph
        self.future_blocks = nn.ModuleList() # the future graph
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            self.history_blocks.append(SSGBlock(hidden_dim=hidden_dim, num_nodes=num_nodes, num_attention_heads=num_attention_heads, topk=topk, nodevec1=self.src_nodevec1, nodevec2=self.src_nodevec2, dropout=dropout))
            self.transition_blocks.append(SSGBlock(hidden_dim=hidden_dim, num_nodes=num_nodes, num_attention_heads=num_attention_heads, topk=topk, nodevec1=self.tgt_nodevec1, nodevec2=self.src_nodevec2, dropout=dropout))
            self.future_blocks.append(SSGBlock(hidden_dim=hidden_dim, num_nodes=num_nodes, num_attention_heads=num_attention_heads, topk=topk, nodevec1=self.tgt_nodevec1, nodevec2=self.tgt_nodevec2, dropout=dropout))
        
        # feed forward
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def feed_forward(self, x):
        x = self.linear2(self.relu(self.dropout(self.linear1(x))))
        return self.dropout(x)
    
    def forward(self, src, tgt):

        # embedding as shown in Equation (14) and (15)
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        for i in range(self.num_blocks):
            src = self.history_blocks[i](src=src, tgt=src) # learning on the history graph, as shown in Equation (16)
        
        for i in range(self.num_blocks):
            tgt = self.transition_blocks[i](src=src, tgt=tgt) # learning on the transition graph, as shown in Equation (17)
        
        for i in range(self.num_blocks):
            tgt = self.future_blocks[i](src=tgt, tgt=tgt) # learning on the future graph, as shown in Equation (19)
        
        tgt = self.feed_forward(tgt) # Equation (20)

        return tgt
