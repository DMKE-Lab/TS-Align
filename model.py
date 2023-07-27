import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedGNN(nn.Module):
    def __init__(self, node_size, rel_size, hidden_size, dropout_rate, depth, device):
        super(ImprovedGNN, self).__init__()
        self.node_size = node_size
        self.rel_size = rel_size
        self.hidden_size = hidden_size
        self.dropout = dropout_rate
        self.depth = depth
        self.device = device

        self.ent_emb = nn.Sequential(
            nn.Linear(node_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.rel_emb = nn.Sequential(
            nn.Linear(rel_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, data):
        adj = data[0]
        ent_ent = data[1]
        ent_rel = data[2]

        adj = adj.to(device=self.device)
        ent_ent = ent_ent.to(device=self.device)
        ent_rel = ent_rel.to(device=self.device)

        he_emb = self.ent_emb(ent_ent)
        hr_emb = self.rel_emb(ent_rel)

        h = torch.cat([he_emb, hr_emb], -1)

        hg_list = []
        hg = h
        for i in range(self.depth - 1):
            h = torch.matmul(ent_ent, h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hg_list.append(h)

        hg_list.append(hg)
        h_mul = torch.cat(hg_list, dim=-1)
        return h_mul







