# ## GCN implementation
#install required libraries
# installation commands changes with plateforms.the following commands can be used for anaconda on windows 10
#conda install -c dglteam dgl
#conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
#pip install networkx

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import time
import csv
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(44298, 16, F.relu)
        self.gcn2 = GCN(16, 4, F.relu)
    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
net = Net()
print("model initialized successfully")

def load_data():
    g = nx.read_edgelist("edge_list.csv", delimiter = ",")
	print("dataset loaded.")
    g = nx.convert_node_labels_to_integers(g)
    g = dgl.DGLGraph(g)
    edge_list = pd.read_csv("edge_list.csv")
    mask = np.load("mask.npy")
    labels = np.load("labels.npy", allow_pickle = True)
    features = np.identity(g.number_of_nodes(), dtype = np.float32)
    return g, th.from_numpy(features), th.from_numpy(labels), mask

g, features, labels, mask = load_data()
mask = th.from_numpy(mask)
labels = labels.type(th.long)
mask = mask.type(th.long)
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []
val_loss = []
epochs = 50
print("now training the model")
for epoch in range(epochs):
    if epoch >=3:
        t0 = time.time()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])
    val_loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch >=3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))
    

fig = plt.figure(figsize=(8,5))
plt.title("GCN validation results")
plt.plot(val_loss, label = "val_loss")
plt.xlabel("number of epochs")
plt.ylabel("")
plt.show()
plt.savefig("val_plot.png")

with open("results_GCN.csv",'w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(val_loss)
    writer.writerow(dur)
    file.close()
    

