import torch
import torch.nn as nn
import torch.nn.functional as F

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LSTM):
    #             for name, param in m.named_parameters():
    #                 if 'weight_ih' in name:
    #                     nn.init.kaiming_uniform_(param, nonlinearity='sigmoid')
    #                 elif 'weight_hh' in name:
    #                     nn.init.orthogonal_(param)
    #                 elif 'bias' in name:
    #                     nn.init.constant_(param, 0)


# class RE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
#         super(RE, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of the latent distribution
#         self.fc_log_var = nn.Linear(hidden_dim, latent_dim)  # Log-variance of the latent distribution
#
#     def forward(self, x, previous_hidden):
#
#         batch_size, seq_len , _ = x.shape
#
#         if previous_hidden is None:
#             h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
#             c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
#         else:
#             h0, c0 = previous_hidden
#
#         lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
#         mu = self.fc_mu(lstm_out[:, -1, :])  # Use the last output for mean
#         log_var = self.fc_log_var(lstm_out[:, -1, :])  # Use the last output for log-variance
#         return mu, log_var, (h_n, c_n)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std


# Linear Decoder
# class Decoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, output_dim):
#         super(Decoder, self).__init__()
#         # print("latent_dim ", latent_dim)
#         self.Linear1 = nn.Linear(latent_dim, hidden_dim)
#         self.act1 = nn.ReLU()
#         self.Linear2 = nn.Linear(hidden_dim, output_dim)
#         self.act2 = nn.Tanh()  # Outputs in range [-1, 1]
#

#
#     def forward(self, x):
#         # print("x ", x.shape)
#         return self.act2(self.Linear2(self.act1(self.Linear1(x))))

class RE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.1):
        super(RE, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1


        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0
                            # , bidirectional=self.bidirectional
                    )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of the latent distribution
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)  # Log-variance of the latent distribution
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        # self.init_weights()

    def forward(self, x, previous_hidden):

        batch_size, seq_len , _ = x.shape

        if previous_hidden is None:
            h0 = torch.zeros(self.num_layers , batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        else:
            h0, c0 = previous_hidden

        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        last_output = lstm_out[:, -1, :]
        last_output = F.dropout(self.ln(last_output), p=self.dropout, training=self.training)
        mu = self.fc_mu(last_output)  # Use the last output for mean
        log_var = self.fc_log_var(lstm_out[:, -1, :])  # Use the last output for log-variance
        return mu, log_var, (h_n, c_n)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout=0.05):
        super(Decoder, self).__init__()
        self.Linear1 = nn.Linear(latent_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.act1 = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.Tanh()

    def forward(self, x):
        h = self.act1(self.ln(self.Linear1(x)))
        # h = F.dropout(h, p=self.dropout, training=self.training)
        return self.act2(self.Linear2(h))

    def freeze_decoder(self):
        for param in self.Linear1.parameters():
            param.requires_grad = False
        for param in self.Linear2.parameters():
            param.requires_grad = False

    def unfreeze_decoder(self):
        for param in self.Linear1.parameters():
            param.requires_grad = True
        for param in self.Linear2.parameters():
            param.requires_grad = True


class StateEncoder(nn.Module):
    def __init__(self, state_dim, hidden_dim, latent_dim):
        super(StateEncoder, self).__init__()
        self.fce1 = nn.Linear(state_dim, hidden_dim)
        self.fce2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = 0.05

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = F.relu(self.fce2(F.relu(self.fce1(x))))
        # h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar



# class StateEncoder(nn.Module):
#     def __init__(self, state_dim, hidden_dim, latent_dim):
#         super(StateEncoder, self).__init__()
#         # Encoder
#         self.fce1 = nn.Linear(state_dim, hidden_dim)
#         self.fce2 = nn.Linear(hidden_dim, hidden_dim)
#         self.ln = nn.LayerNorm(hidden_dim)
#         self.dropout = 0.05
#
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         h = F.relu(self.fce2(F.relu(self.fce1(x))))
#         h = F.dropout(self.ln(h), p=self.dropout, training=self.training)
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class RE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.1):
#         super(RE, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         self.bidirectional = True
#         self.num_directions = 2 if self.bidirectional else 1
#
#         self.lstm = nn.LSTM(
#             input_dim,
#             hidden_dim,
#             num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0,
#             bidirectional=self.bidirectional
#         )
#
#         # NOTE: updated to match bidirectional output dimension
#         self.fc_mu = nn.Linear(hidden_dim * self.num_directions, latent_dim)
#         self.fc_log_var = nn.Linear(hidden_dim * self.num_directions, latent_dim)
#         self.ln = nn.LayerNorm(hidden_dim * self.num_directions)
#         self.dropout = dropout
#
#     def forward(self, x, previous_hidden):
#         batch_size, seq_len, _ = x.shape
#
#         if previous_hidden is None:
#             h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, device=x.device)
#             c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, device=x.device)
#         else:
#             h0, c0 = previous_hidden
#
#         lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
#         last_output = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim * 2)
#         last_output = F.dropout(self.ln(last_output), p=self.dropout, training=self.training)
#
#         mu = self.fc_mu(last_output)
#         log_var = self.fc_log_var(last_output)
#
#         return mu, log_var, (h_n, c_n)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#
# class Decoder(nn.Module):
#     def __init__(self, latent_dim, hidden_dim, output_dim):
#         super(Decoder, self).__init__()
#         self.Linear1 = nn.Linear(latent_dim, hidden_dim)  # NOTE: assumes concatenated latent vectors
#         self.act1 = nn.ReLU()
#         self.Linear2 = nn.Linear(hidden_dim, output_dim)
#         self.act2 = nn.Tanh()  # Outputs in range [-1, 1]
#
#     def freeze_decoder(self):
#         for param in self.parameters():
#             param.requires_grad = False
#
#     def unfreeze_decoder(self):
#         for param in self.parameters():
#             param.requires_grad = True
#
#     def forward(self, x):
#         return self.act2(self.Linear2(self.act1(self.Linear1(x))))
#
#
# class StateEncoder(nn.Module):
#     def __init__(self, state_dim, hidden_dim, latent_dim):
#         super(StateEncoder, self).__init__()
#         self.fce1 = nn.Linear(state_dim, hidden_dim)
#         self.fce2 = nn.Linear(hidden_dim, hidden_dim)
#
#         # NOTE: assuming Decoder expects concatenated mu and z, hence 2*latent_dim
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         h = F.relu(self.fce2(F.relu(self.fce1(x))))
#         mu = self.fc_mu(h)
#         logvar = self.fc_logvar(h)
#         return mu, logvar
