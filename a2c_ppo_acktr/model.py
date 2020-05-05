import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, use_pnn=False):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if use_pnn:
                base = PNNConvBase
            elif len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


##### Added for CCM  #######

class ScaleLayer(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class PNNBase(NNBase):
    def __init__(self, t, recurrent=False, hidden_size=512):
        super(PNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(t[0][0], t[0][1], t[0][2], stride=t[0][3]))
        self.conv2 = init_(nn.Conv2d(t[1][0], t[1][1], t[1][2], stride=t[1][3]))
        self.conv3 = init_(nn.Conv2d(t[2][0], t[2][1], t[2][2], stride=t[2][3]))
        self.fc = init_(nn.Linear(t[3][0], t[3][1]))

        self.mp = None

        self.relu = nn.ReLU()
        self.flatten = Flatten()

        self.topology = [
            [t[1][2], t[1][3]],
            [t[2][2], t[2][3]],
            t[3][1]
        ]

        self.output_shapes = [x[1] for x in t]
        self.input_shapes = [x[0] for x in t]

    def layers(self, i, x):
        if i == 0:
            if not self.mp:
                return self.relu(self.conv1(x))
            else:
                return self.mp(self.relu(self.conv1(x)))
        elif i == 1:
            return self.relu(self.conv2(x))
        elif i == 2:
            return self.relu(self.conv3(x))
        elif i == 3:
            return self.fc(self.flatten(x))

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.layers(i, x)
            outs.append(x)

        return outs


class PNNColumnAtari(PNNBase):  # Use this for atari environments
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        t = [[num_inputs, 32, 8, 4], [32, 64, 4, 2], [64, 32, 3, 1], [32 * 7 * 7, hidden_size]]
        # [n_input, n_output, fsize, stride] for c1, c2, c3 and [n_input, n_output] for FC

        super(PNNColumnAtari, self).__init__(t, recurrent, hidden_size)


class PNNColumnGrid(PNNBase):  # Use this for grid environments
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        t = [[num_inputs, 16, 2, 1], [16, 32, 2, 1], [32, 64, 2, 1], [64, 64]]

        super(PNNColumnGrid, self).__init__(t, recurrent, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.mp = nn.MaxPool2d((2, 2))
        self.fc  = nn.Sequential(
            init_(nn.Linear(hidden_size, 64)),
            nn.Tanh(),
            self.fc
        )


class PNNConvBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, grid=False, hidden_size=512):
        super(PNNConvBase, self).__init__(recurrent, hidden_size, hidden_size)
        self.columns = nn.ModuleList([])
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.recurrent = recurrent
        self.alpha = nn.ModuleList([])
        self.V = nn.ModuleList([])
        self.U = nn.ModuleList([])
        self.flatten = Flatten()
        self.grid = grid

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        if grid:
            self.critic_linear = nn.Sequential(
                init_(nn.Linear(self.hidden_size, 64)),
                nn.Tanh(),
                init_(nn.Linear(64, 1))
            )
        else:
            self.critic_linear = init_(nn.linear(self.hidden_size,1))
        

        self.train()
        self.n_layers = 4

    def forward(self, x, rnn_hxs, masks):
        assert self.columns, 'PNN should at least have one column (missing call to `new_task` ?)'
        # x = (x / 255.0)

        inputs = [self.columns[i].layers(0, x) for i in range(len(self.columns))]

        for l in range(1, self.n_layers):
            outputs = [self.columns[0].layers(l, inputs[0])]
            for c in range(1, len(self.columns)):

                pre_col = inputs[c - 1]

                cur_out = self.columns[c].layers(l, inputs[c])

                a = self.alpha[c - 1][l - 1]
                a_h = F.relu(a(pre_col))

                V = self.V[c - 1][l - 1]
                V_a_h = F.relu(V(a_h))

                U = self.U[c - 1][l - 1]

                if l == self.n_layers - 1:  # FC layer
                    V_a_h = self.flatten(V_a_h)
                    U_V_a_h = U(V_a_h)
                    out = F.relu(cur_out + U_V_a_h)
                    outputs.append(out)

                else:
                    U_V_a_h = U(V_a_h)  # conv layers
                    out = F.relu(cur_out + U_V_a_h)
                    outputs.append(out)

            inputs = outputs

        x = inputs[-1]
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

    def new_task(self):  # adds a new column to pnn
        if self.grid:
            new_column = PNNColumnGrid(self.num_inputs, self.recurrent, self.hidden_size)
        else:
            new_column = PNNColumnAtari(self.num_inputs, self.recurrent, self.hidden_size)

        self.columns.append(new_column)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        if len(self.columns) > 1:

            pre_col, col = self.columns[-2], self.columns[-1]

            a_list = []
            V_list = []
            U_list = []
            for l in range(1, self.n_layers):
                a = ScaleLayer(0.01)

                map_in = pre_col.output_shapes[l - 1]
                map_out = int(map_in / 2)
                v = init_(nn.Conv2d(map_in, map_out, 1))

                if l != self.n_layers - 1:  # conv -> conv, last layer

                    cur_out = col.output_shapes[l]
                    size, stride = pre_col.topology[l - 1]
                    u = init_(nn.Conv2d(map_out, cur_out, size, stride=stride))

                else:
                    input_size = int(col.input_shapes[-1] / 2)
                    hidden_size = self.hidden_size

                    u = init_(nn.Linear(input_size, hidden_size))

                a_list.append(a)
                V_list.append(v)
                U_list.append(u)

            a_list = nn.ModuleList(a_list)
            V_list = nn.ModuleList(V_list)
            U_list = nn.ModuleList(U_list)

            self.alpha.append(a_list)
            self.V.append(V_list)
            self.U.append(U_list)

    def freeze_columns(self, skip=None):  # freezes the weights of previous columns
        if skip is None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

    def parameters(self, col=None):
        if col is None:
            return super(PNNConvBase, self).parameters()

        return self.columns[col].parameters()


