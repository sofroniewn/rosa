import torch
import torch.nn as nn


class BilinearHead(nn.Module):
    def __init__(
        self,
        in_dim_1,
        in_dim_2,
        rank,
        head_1="Linear",
        head_2="Linear",
    ):
        super(BilinearHead, self).__init__()
        self.fc1 = SingleHead(in_dim_1, rank, head=head_1)
        self.fc2 = SingleHead(in_dim_2, rank, head=head_2)
        self.act = nn.Softplus()

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = (x1 * x2).sum(-1)
        x = self.act(x)
        return x


class ConcatHead(nn.Module):
    def __init__(
        self,
        in_dim_1,
        in_dim_2,
        head="Linear",
    ):
        super(ConcatHead, self).__init__()
        self.fc = SingleHead(in_dim_1 + in_dim_2, 1, head=head)
        self.act = nn.Softplus()

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), -1)
        x = self.fc(x)
        x = self.act(x)
        return x


class SingleHead(nn.Module):
    def __init__(self, in_dim, out_dim, head="Linear"):
        super(SingleHead, self).__init__()
        self.head = head
        if self.head == "Linear":
            self.model = nn.Linear(in_features=in_dim, out_features=out_dim)
        elif self.head == "MLP":
            self.model = MLP(in_dim=in_dim, out_dim=out_dim, dropout=0.5)
        elif self.head == "OneHot":
            self.model = nn.Embedding(in_dim, out_dim)
        elif self.head == "Identity":
            self.model = nn.Identity()
        else:
            raise ValueError(f"Model type {self.head} not recognized")

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, dropout=0.0, in_dim=512, out_dim=512, hidden_dim=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=True)
        self.act1 = nn.Softplus()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.act2 = nn.Softplus()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
