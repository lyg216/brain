import argparse
import torch
import torch.nn.functional as F
from networks import GNNs
from dynamicGnn import *
from torch import tensor
from torch.optim import Adam
import numpy as np
import os,random
from torch_geometric.data import Data, Batch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, GINConv,SAGEConv,GraphConv,TransformerConv,ResGatedGraphConv,ChebConv,GATConv,SGConv,GeneralConv
from torch_geometric.loader import DataLoader
import os.path as osp
from utils import *
import sys
import time
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DynHCPGender')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--x', type=str, default="corr")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="TransformerConv")
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=32)
parser.add_argument('--num_heads', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=200)  # 增加训练轮数
parser.add_argument('--echo_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()

path = "base_params/"
res_path = "base_results/"
path_data = "../../data/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(res_path):
    os.mkdir(res_path)

def logger(info):
    f = open(os.path.join(res_path, 'dynamic_results.csv'), 'a')
    print(info, file=f)

log = "dataset,model,hidden,num_layers,epochs,batch size,loss,acc,std"
logger(log)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# 图增强函数
def augment_graph(data):
    edge_index = data.edge_index
    x = data.x
    
    # 随机删除边
    num_edges = edge_index.size(1)
    num_keep = int(num_edges * 0.8)  # 保留80%的边
    perm = torch.randperm(num_edges)[:num_keep]
    edge_index = edge_index[:, perm]
    
    # 添加特征噪声
    x = x + torch.randn_like(x) * 0.1
    
    return Data(x=x, edge_index=edge_index, y=data.y)

# 修改 DynamicGNN 类
class DynamicGNN(torch.nn.Module):
    def __init__(self, num_features, hidden1, hidden2, num_heads, num_layers, gnn, dropout, num_classes, proj_dim=64):
        super(DynamicGNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(gnn(num_features, hidden1, heads=num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(gnn(hidden1 * num_heads, hidden1, heads=num_heads))
        self.convs.append(gnn(hidden1 * num_heads, hidden2, heads=1))
        self.fc = torch.nn.Linear(hidden2, num_classes)
        self.dropout = dropout
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hidden2, proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_repr = x
        x = self.fc(x)
        return F.log_softmax(x, dim=1), node_repr

    def project(self, node_repr):
        return self.projector(node_repr)

# 对比损失函数
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return loss / 2

start = time.time()
if args.dataset=='DynHCPGender':
    dataset_raw = torch.load(os.path.join(path_data,args.dataset,"processed", args.dataset+".pt"))
    dataset,labels = [],[]
    for v in dataset_raw:
        batches = v.get('batches')
        if len(batches)>0:
            for b in batches:
                y = b.y[0].item()
                dataset.append(b)
                labels.append(y)
else:
    dataset = torch.load(os.path.join(path_data,args.dataset,"processed", args.dataset+".pt"))
    labels = dataset['labels']
    dataset = dataset['batches']

print("dataset loaded successfully!",args.dataset)

train_tmp, test_indices = train_test_split(list(range(len(labels))),
                        test_size=0.2, stratify=labels,random_state=123,shuffle= True)

tmp = [dataset[i] for i in train_tmp]
labels_tmp = [labels[i] for i in train_tmp]
train_indices, val_indices = train_test_split(list(range(len(labels_tmp))),
test_size=0.125, stratify=labels_tmp,random_state=123,shuffle = True)
train_dataset = [tmp[i] for i in train_indices]
val_dataset = [tmp[i] for i in val_indices]
train_labels= [labels_tmp[i] for i in train_indices]
val_labels = [labels_tmp[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]
test_labels =[labels[i] for i in test_indices]

print("dataset {} loaded with train {} val {} test {} splits".format(args.dataset,len(train_dataset), len(val_dataset), len(test_dataset)))

args.num_features,args.num_classes = 100,len(np.unique(labels))
print("number of features and classes",args.num_features,args.num_classes)
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:  
        data = data.to(args.device)
        aug_data = augment_graph(data).to(args.device)
        optimizer.zero_grad()
        out, node_repr = model(data)
        aug_out, aug_node_repr = model(aug_data)
        
        z1 = model.project(node_repr)
        z2 = model.project(aug_node_repr)
        
        cls_loss = criterion(out, data.y)
        con_loss = contrastive_loss(z1, z2)
        
        loss = cls_loss + 0.5 * con_loss  # 可以调整对比损失的权重
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    correct = 0
    for data in loader:  
        data = data.to(args.device)
        with torch.no_grad():
            out, _ = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    return correct / len(loader)

seeds = [123,124]
for index in range(args.runs):
    gnn = eval(args.model)
    model = DynamicGNN(args.num_features,args.hidden1,args.hidden2,args.num_heads,args.num_layers,gnn,args.dropout, args.num_classes).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss, test_acc = [],[]
    best_val_acc,best_val_loss,pat = 0.0,0.0,0
    for epoch in range(args.epochs):
        ep_start = time.time()
        loss = train(train_dataset)
        val_acc = test(val_dataset)
        test_acc = test(test_dataset)
        if epoch%10==0:
            print("epoch: {}, loss: {}, val_acc:{}, test_acc:{}".format(epoch, np.round(loss,6), np.round(val_acc,2),np.round(test_acc,2)))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            pat = 0 
            torch.save(model.state_dict(), path + args.dataset+args.model+'-checkpoint-best-acc.pkl')
        else:
            pat += 1
        if pat >=args.early_stopping and epoch > args.epochs // 2:
            print("early stopped!")
            break
        ep_end = time.time()
        print("epoch time:", ep_end-ep_start)
    model.load_state_dict(torch.load(path + args.dataset+args.model+'-checkpoint-best-acc.pkl'))
    model.eval()
    test_acc = test(test_dataset)
    test_loss = train(test_dataset)

    log = "{},{},{},{},{},{},{},{},{}".format(args.dataset,args.model,args.hidden1,args.num_layers,args.epochs,args.batch_size,np.round(test_loss,4),np.round(test_acc,4),np.round(np.std(test_acc),4))
    logger(log)