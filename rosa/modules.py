


class BilinearHead(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, n_dim_1, n_dim_2, rank, bias=False, head_1='Linear', head_2='Linear'):
        super(BilinearHead, self).__init__()
        self.bias = bias
        if head_1 == 'OneHot':
            in_dim_1 = n_dim_1
        else:
            in_dim_1 = in_dim_1
        if head_2 == 'OneHot':
            in_dim_2 = n_dim_2
        else:
            in_dim_2 = in_dim_2
        if head_1 == 'Identity':
            out_dim_2 = in_dim_1
        else:
            out_dim_2 = rank
        if head_2 == 'Identity':
            out_dim_1 = in_dim_2
        else:
            out_dim_1 = rank
        self.fc1 = SingleHead(in_dim_1, out_dim_1, head=head_1)
        self.fc2 = SingleHead(in_dim_2, out_dim_2, head=head_2)
        if self.bias:
            self.bc1 = SingleHead(in_dim_1, 1, head=head_1)
            self.bc2 = SingleHead(in_dim_2, 1, head=head_2)

        self.act = nn.Softplus()

    def forward(self, x):
        idx1, idx2, x1, x2, = x
        if self.fc1.head == 'OneHot':
            x1 = idx1
        if self.fc2.head == 'OneHot':
            x2 = idx2
        x1r = self.fc1(x1)
        x2r = self.fc2(x2)
        x = (x1r * x2r).sum(-1)
        if self.bias:
            x1b = self.bc1(x1)
            x2b = self.bc2(x2)
            x += x1b.reshape(x.shape) + x2b.reshape(x.shape)
        x = self.act(x)
        return x


class ConcatHead(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, n_dim_1, n_dim_2, rank, bias=False, head_1='Linear', head_2='Linear'):
        super(ConcatHead, self).__init__()
        self.bias = bias
        if head_1 == 'OneHot':
            in_dim_1 = n_dim_1
        else:
            in_dim_1 = in_dim_1
        if head_2 == 'OneHot':
            in_dim_2 = n_dim_2
        else:
            in_dim_2 = in_dim_2
        if head_1 == 'Identity':
            out_dim_2 = in_dim_1
        else:
            out_dim_2 = rank
        if head_2 == 'Identity':
            out_dim_1 = in_dim_2
        else:
            out_dim_1 = rank
        self.fc = SingleHead(in_dim_1 + in_dim_2, 1, head=head_1)
        if self.bias:
            self.bc1 = SingleHead(in_dim_1, 1, head=head_1)
            self.bc2 = SingleHead(in_dim_2, 1, head=head_2)

        self.act = nn.Softplus()

    def forward(self, x):
        idx1, idx2, x1, x2, = x
        x = torch.concat((x1, x2), -1)
        x = self.fc(x).reshape(idx1.shape)
        if self.bias:
            x1b = self.bc1(x1)
            x2b = self.bc2(x2)
            x += x1b.reshape(x.shape) + x2b.reshape(x.shape)
        x = self.act(x)
        return x


class SingleHead(nn.Module):
    def __init__(self, in_dim, out_dim, head='Linear'):
        super(SingleHead, self).__init__()
        self.head = head
        if self.head == 'Linear':
            self.model = nn.Linear(in_features=in_dim, out_features=out_dim)
        elif self.head == 'MLP':
            self.model = MLP(in_dim=in_dim, out_dim=out_dim, dropout=0.5)
        elif self.head == 'OneHot':
            self.model = nn.Embedding(in_dim, out_dim)
        elif self.head == 'Identity':
            self.model = nn.Identity()
        else:
            raise ValueError(f'Model type {self.head} not recognized')

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, dropout=0., in_dim=512, out_dim=10, hidden_dim=1024):
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