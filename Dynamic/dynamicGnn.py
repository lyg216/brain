import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import math

softmax = torch.nn.LogSoftmax(dim=1)

def augment_graph(batch):
    augmented_batch = []
    for data in batch:
        edge_index = data.edge_index
        x = data.x
        
        # 随机删除边
        num_edges = edge_index.size(1)
        num_keep = int(num_edges * 0.8)  # 保留80%的边
        perm = torch.randperm(num_edges)[:num_keep]
        edge_index = edge_index[:, perm]
        
        # 添加特征噪声
        x = x + torch.randn_like(x) * 0.1
        
        augmented_batch.append(data.__class__(x=x, edge_index=edge_index, y=data.y))
    return augmented_batch

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return loss / 2

class DynamicGNN(nn.Module):
    def __init__(self, dataset, input_dim, hidden_channels, hidden_dim, num_heads, num_layers, GNN, dropout, num_classes, k=0.6, proj_dim=64):
        super(DynamicGNN, self).__init__()
        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)
        self.aggr = aggr.SumAggregation()
        self.convs = ModuleList()
        self.convs.append(GNN(dataset.num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.hidden_dim = hidden_dim
        self.multihead_attn = nn.MultiheadAttention(hidden_channels, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.layer_norm2 = nn.LayerNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_channels))
        self.linear = nn.Linear(hidden_channels, num_classes)
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def forward(self, batch):
        xs = []
        for b in batch:        
            x = b.x
            for conv in self.convs:
                x = conv(x, b.edge_index).tanh()
            x = self.aggr(x)
            xs.append(x)
        x = torch.stack(xs, dim=0)
        x = x.squeeze(dim=1)
        x, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x)
        x = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        x = x_attend.relu()
        x = torch.sum(x, dim=0)
        node_repr = x
        x = self.linear(x)
        return x, node_repr

    def project(self, node_repr):
        return self.projector(node_repr)

def train(model, optimizer, criterion, batch, device):
    model.train()
    optimizer.zero_grad()
    
    aug_batch = augment_graph(batch)
    batch = [b.to(device) for b in batch]
    aug_batch = [b.to(device) for b in aug_batch]
    
    logits, node_repr = model(batch)
    aug_logits, aug_node_repr = model(aug_batch)
    
    z1 = model.project(node_repr)
    z2 = model.project(aug_node_repr)
    
    cls_loss = criterion(logits, batch[0].y)
    con_loss = contrastive_loss(z1, z2)
    
    loss = cls_loss + 0.5 * con_loss  # 可以调整对比损失的权重
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test(model, batch, device):
    model.eval()
    with torch.no_grad():
        batch = [b.to(device) for b in batch]
        logits, _ = model(batch)
        pred = logits.argmax(dim=1)
        correct = int((pred == batch[0].y).sum())
    return correct, len(batch[0].y)