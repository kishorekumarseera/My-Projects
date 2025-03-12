import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
from scipy.sparse import coo_matrix

nodes=[]
labels=[]
attributes=[]

mydict={"Case_Based":0,"Genetic_Algorithms":1,"Neural_Networks":2,"Probabilistic_Methods":3,
        "Reinforcement_Learning":4,"Rule_Learning":5,"Theory":6}
mydict1={}
with open("../cora.content", "r") as file:
    # Read the entire contents of the file
    for line in file:
        line = line.strip()
        line = line.split('\t')
        nodes.append(line[0])
        labels.append(mydict[line[-1]])
        attributes.append([line[1:-1]])
        mydict1[line[0]]=mydict[line[-1]]

attributes = np.array(attributes, dtype=np.float32)

X_train=[]
y_train=[]
edges=[]
with open("../cora_train.cites", "r") as file:
    # Read the entire contents of the file
    for line in file:
        line = line.strip()
        line = line.split('\t')
        X_train.append(line[1])
        y_train.append(mydict1[line[1]])
        edges.append([line[1],line[0]])
X_test=[]
y_test=[]
with open("../cora_test.cites", "r") as file:
    # Read the entire contents of the file
    for line in file:
        line = line.strip()
        line = line.split('\t')
        X_test.append(line[1])
        y_test.append(mydict1[line[1]])
        edges.append([line[1],line[0]])

node_ind={}
for i in range(len(nodes)):
    node_ind[nodes[i]]=i

index_row = []
index_col = []
values = []

for i, j in edges:
    row_index = node_ind.get(i)
    col_index = node_ind.get(j)
    if row_index is not None and col_index is not None:
        index_row.append(row_index)
        index_col.append(col_index)
        values.append(1)
n = len(nodes)
adjacency_matrix = coo_matrix((values, (index_row, index_col)), shape=(n, n), dtype=np.int8)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Transpose weight matrix before multiplication
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_nodes):
        super(GCN, self).__init__()
        self.num_nodes = num_nodes
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # Add a reshape operation to ensure x is formatted as a matrix
        x = x.view(-1, x.size(1))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)



def normalize_adj(adj):
    """Normalize adjacency matrix."""
    adj = adj + torch.eye(adj.size(0))  # Add self-loops
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)

def train_model(model, optimizer, criterion, x, adj, idx_train, y_train_tensor):
    """Train the GCN model."""
    model.train()
    optimizer.zero_grad()
    output = model(x, adj)
    train_output = output[idx_train]
    loss = criterion(train_output, y_train_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, x, adj, idx_test, y_test_tensor):
    """Evaluate the GCN model."""
    model.eval()
    output = model(x, adj)
    test_output = output[idx_test]
    pred_labels = test_output.argmax(dim=1).cpu().numpy()
    true_labels = y_test_tensor.cpu().numpy()
    report = classification_report(true_labels, pred_labels)
    with open("gcn_metrics.txt", "w") as f:
        f.write(report)

# Convert data to PyTorch tensors
adj = torch.FloatTensor(adjacency_matrix.todense())  # Convert adjacency matrix to dense tensor
adj = normalize_adj(adj)  # Normalize adjacency matrix


x = torch.FloatTensor(attributes)
x = x.squeeze()
y = torch.LongTensor(labels)
idx_train = torch.tensor([node_ind[node_id] for node_id in X_train], dtype=torch.long)
idx_test = torch.tensor([node_ind[node_id] for node_id in X_test], dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Model initialization and optimization
num_nodes = len(nodes)
model = GCN(nfeat=x.shape[1], nhid=16, nclass=len(set(labels)), dropout=0.5, num_nodes=num_nodes)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(2000):
    loss = train_model(model, optimizer, criterion, x, adj, idx_train, y_train_tensor)
    print(f'Epoch: {epoch+1}, Loss: {loss}')

# Evaluation
evaluate_model(model, x, adj, idx_test, y_test_tensor)