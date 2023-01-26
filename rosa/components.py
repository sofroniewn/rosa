import torch
import torch.nn as nn


class BilinearHead(nn.Module):
    def __init__(
        self,
        in_dim_1,
        in_dim_2,
        rank,
        head_1="linear",
        head_2="linear",
    ):
        super(BilinearHead, self).__init__()
        self.fc1 = SingleHead(in_dim_1, rank, head=head_1)
        self.fc2 = SingleHead(in_dim_2, rank, head=head_2)
        self.act = nn.Softplus()

    def forward(self, x):
        x1, x2 = x
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = (x1 * x2).sum(-1)
        # x = self.act(x)
        return x


class ConcatHead(nn.Module):
    def __init__(
        self,
        in_dim_1,
        in_dim_2,
        head="linear",
    ):
        super(ConcatHead, self).__init__()
        self.fc = SingleHead(in_dim_1 + in_dim_2, 1, head=head)
        self.act = nn.Softplus()

    def forward(self, x):
        x1, x2 = x
        shape = list(x1.shape)
        shape[-1] += x2.shape[-1]
        x = torch.concat((x1, x2), -1)
        x = self.fc(x).reshape(shape[:-1])
        # x = self.act(x)
        return x


class SingleHead(nn.Module):
    def __init__(self, in_dim, out_dim, head="linear"):
        super(SingleHead, self).__init__()
        self.head = head
        self.out_dim = out_dim
        if self.head == "linear":
            self.fc = nn.Linear(in_features=in_dim, out_features=out_dim)
        elif self.head == "MLP":
            self.fc = MLP(in_dim=in_dim, out_dim=out_dim, dropout=0.5)
        elif self.head == "OneHot":
            self.fc = nn.Embedding(in_dim, out_dim)
        elif self.head == "Identity":
            self.fc = nn.Identity()
        else:
            raise ValueError(f"Model type {self.head} not recognized")
        
        # self.act1 = nn.ReLU()
        # self.fc2 = nn.Linear(out_dim, 128 * out_dim)
        # self.norm = nn.BatchNorm1d(out_dim)
        # self.act = nn.Softplus()
        # self.act = nn.Sigmoid()
        # self.library_size = nn.Parameter(torch.tensor(0.5))
        # self.act = nn.Softmax(dim=-1)
        # self.mult = nn.Parameter(torch.tensor(0.7*1e5))
        # self.mult = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.fc(x)
        # x = self.act1(x)
        # x = self.norm(x)
        # x = self.fc2(x)
        # x = x.view(-1, 256, self.out_dim)
        # x = torch.exp(x)
        # x = self.act(x)
        # # x = x * x.shape[0] * self.mult
        # x = x * self.mult
        # x = torch.log1p(x)
        return x


class MLP(nn.Module):
    def __init__(self, dropout=0.0, in_dim=512, out_dim=512, hidden_dim=128):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=True),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=out_dim, bias=True)
        )

    def forward(self, x):
        return self.network(x)
