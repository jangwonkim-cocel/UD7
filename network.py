import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weight_init, AvgL1Norm


class EnsembleQNet(nn.Module):
    def __init__(self, num_critics, state_dim, action_dim, device, zs_dim=256, hidden_dims=(256, 256), activation_fc=F.elu):
        super(EnsembleQNet, self).__init__()
        self.device = device
        self.activation_fc = activation_fc

        self.num_critics = num_critics

        self.q_nets = nn.ModuleList()
        for _ in range(self.num_critics):
            q_net = self._build_q_net(state_dim, action_dim, zs_dim, hidden_dims)
            self.q_nets.append(q_net)

        self.apply(weight_init)

    def _build_q_net(self, state_dim, action_dim, zs_dim, hidden_dims):
        q_net = nn.ModuleDict({
            's_input_layer': nn.Linear(state_dim + action_dim, hidden_dims[0]),
            'emb_input_layer': nn.Linear(2 * zs_dim + hidden_dims[0], hidden_dims[0]),
            'emb_hidden_layers': nn.ModuleList([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
            ]),
            'output_layer': nn.Linear(hidden_dims[-1], 1)
        })
        return q_net

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action, zsa, zs):
        s, a = self._format(state, action)
        sa = torch.cat([s, a], dim=1)
        embeddings = torch.cat([zsa, zs], dim=1)

        q_values = []
        for q_net in self.q_nets:
            q = AvgL1Norm(q_net['s_input_layer'](sa))
            q = torch.cat([q, embeddings], dim=1)
            q = self.activation_fc(q_net['emb_input_layer'](q))
            for hidden_layer in q_net['emb_hidden_layers']:
                q = self.activation_fc(hidden_layer(q))
            q = q_net['output_layer'](q)
            q_values.append(q)

        return torch.cat(q_values, dim=1)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, device, zs_dim=256, hidden_dims=(256, 256), activation_fc=F.relu):
        super(Policy, self).__init__()
        self.device = device
        self.apply(weight_init)

        self.activation_fc = activation_fc

        self.s_input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.zss_input_layer = nn.Linear(zs_dim + hidden_dims[0], hidden_dims[0])
        self.zss_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.zss_hidden_layers.append(hidden_layer)
        self.zss_output_layer = nn.Linear(hidden_dims[-1], action_dim)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        return x

    def forward(self, state, zs):
        state = self._format(state)

        state = AvgL1Norm(self.s_input_layer(state))
        zss = torch.cat([state, zs], 1)

        zss = self.activation_fc(self.zss_input_layer(zss))
        for i, hidden_layer in enumerate(self.zss_hidden_layers):
            zss = self.activation_fc(hidden_layer(zss))
        zss = self.zss_output_layer(zss)
        action = torch.tanh(zss)
        return action


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, device, zs_dim=256, hidden_dims=(256, 256), activation_fc=F.elu):
        super(Encoder, self).__init__()
        self.device = device
        self.activation_fc = activation_fc

        self.s_encoder_input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.s_encoder_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.s_encoder_hidden_layers.append(hidden_layer)
        self.s_encoder_output_layer = nn.Linear(hidden_dims[-1], zs_dim)

        self.zsa_encoder_input_layer = nn.Linear(zs_dim + action_dim, hidden_dims[0])
        self.zsa_encoder_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.zsa_encoder_hidden_layers.append(hidden_layer)
        self.zsa_encoder_output_layer = nn.Linear(hidden_dims[-1], zs_dim)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def zs(self, state):
        state = self._format(state)

        zs = self.activation_fc(self.s_encoder_input_layer(state))
        for i, hidden_layer in enumerate(self.s_encoder_hidden_layers):
            zs = self.activation_fc(hidden_layer(zs))
        zs = AvgL1Norm(self.s_encoder_output_layer(zs))
        return zs

    def zsa(self, zs, action):
        action = self._format(action)
        zsa = torch.cat([zs, action], 1)

        zsa = self.activation_fc(self.zsa_encoder_input_layer(zsa))
        for i, hidden_layer in enumerate(self.zsa_encoder_hidden_layers):
            zsa = self.activation_fc(hidden_layer(zsa))
        zsa = self.zsa_encoder_output_layer(zsa)
        return zsa
