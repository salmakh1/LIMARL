import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class SLRNNAgent(nn.Module):
    def __init__(self, input_shape, hidden_shape, args):
        super(SLRNNAgent, self).__init__()
        self.args = args
        self.fc_sl = nn.Linear(hidden_shape, args.rnn_hidden_dim//2)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim//2)
        self.fc_conc = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, mu, hidden_state):
        b, a, e = inputs.size()
        # mu = mu.reshape(b, a, -1)
        _, e_sl = mu.size()



        inputs = inputs.view(-1, e)
        # mu = mu.view(-1, e_sl)

        mu_ = F.relu(self.fc_sl(mu), inplace=True)
        x_ = F.relu(self.fc1(inputs), inplace=True)

        # Concatenate x_ and mu_ along the feature dimension (dim=1)
        x = th.cat((x_, mu_), dim=1)
        # x = F.relu(self.fc_conc(x), inplace=True)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)