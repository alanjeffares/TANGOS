import torch
import torch.nn as nn

class UCI_MLP(nn.Module):
    def __init__(self, num_features, num_outputs, dropout=0, batch_norm=False):
        super(UCI_MLP, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.batch_norm = batch_norm
        d = num_features + 1
        self.fc1 = nn.Linear(num_features, d)
        self.bn1 = nn.BatchNorm1d(d)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(d)
        self.relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(d, num_outputs)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.fc1(x)
        if self.batch_norm and batch_size > 1:
          out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.batch_norm and batch_size > 1:
          out = self.bn2(out)
        h_output = self.relu2(out)
        h_output = self.dropout(h_output)
        out = self.fc3(h_output)
        return out, h_output