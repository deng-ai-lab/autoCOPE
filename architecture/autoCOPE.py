import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from architecture.preprocessing_unit_lib import unit_lib
import numpy as np

class AUTOCOPE(nn.Module):
    def __init__(self, args, device='cpu'):
        super(AUTOCOPE, self).__init__()
        self.lib = unit_lib(args)
        self.device = device

        # 1. Define properties of the preprocessing schemes.
        self.moi = False
        self.max_size = args.graph_max_size
        if self.moi == True:
            self.num_modality = 2
        else:
            self.num_modality = 1

        # 2. Define the shape of the scPSAD model.
        self.embedding_size = args.embedding_size
        self.len_std = len(self.lib.nodes_std)
        self.len_moi = len(self.lib.nodes_moi)
        # self.len_dr = len(self.lib.nodes_dr)
        self.hidden_size = args.hidden_size
        if self.moi == True:
            len_action = self.len_std + self.len_moi
        else:
            len_action = self.len_std

        # 3. Define components of the scPSAD architecture.
        # 3.1. Layer for embedding action types, that is shared across pan-preprocessing-scheme.
        self.embedding = nn.Embedding(len_action, self.embedding_size)
        self.rnn = nn.LSTMCell(self.embedding_size, self.hidden_size)

        # 3.2. Layer for decoding standard preprocessing policy, that is shared across modality. Units include:
        #    1) feature selection operations: Top HVDs with 9 choices of parameters.
        #    2) scaling operations: 2 operations frequently used in NGS including log and standardization, 5
        #       S-type curves that do not influence the expression pattern including sqrt, alf, erf, gd and tanh.
        #    3) normalization operations: 8 operations frequently used in NGS.
        #    4) stop: flag to stop the generation of the standard chain.
        self.std_decoder = nn.Linear(self.hidden_size, self.len_std)

        # 3.3. Layer for decoding dimensionality reduction policy, that is shared across modality:
        #      5 types * 9 choices of parameters = 45 types.
        # self.dr_decoder = nn.Linear(self.hidden_size, self.len_dr)

        # 3.4. Layer for decoding multi-omics integration policy: 5 types.
        self.moi_decoder = nn.Linear(self.hidden_size, self.len_moi)

        # 4. Initialize the parameters.
        self.init_parameters()

    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.std_decoder.bias.data.fill_(0)
        # self.dr_decoder.bias.data.fill_(0)
        self.moi_decoder.bias.data.fill_(0)

    def init_hidden(self):
        h_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)
        c_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)

        return (h_t, c_t)

    def forward(self, input, h_t, c_t, decoder):
        input = self.embedding(input)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        logits = decoder(h_t)
        return h_t, c_t, logits

    def sample(self):
        input = torch.LongTensor([0]).to(self.device)
        h_t, c_t = self.init_hidden()
        actions_p = []
        actions_log_p = []
        actions_index = []
        len_std_per_modality = [0 for i in range(self.num_modality)]

        # 1. Automatic generation of the standard preprocessing chain part that is specific to different modalities.
        ps_ref_embedding, ps_target_embedding = None, None
        ps_embedding = []
        for modality in range(self.num_modality):
            item = None
            while (item != 'stop' and len_std_per_modality[modality] < self.max_size):
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.std_decoder)
                action_index = Categorical(logits=logits).sample()
                p = F.softmax(logits, dim=-1)[0,action_index]
                log_p =F.log_softmax(logits, dim=-1)[0,action_index]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)

                input = action_index
                print(self.lib.nodes_std)
                print(action_index)
                item = self.lib.nodes_std[action_index]
                len_std_per_modality[modality] += 1
            ps_embedding.append(c_t.detach())

        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)
        actions_index = torch.cat(actions_index)
        ps_embedding = torch.cat(ps_embedding)

        return actions_p, actions_log_p, actions_index, ps_embedding, len_std_per_modality

    def get_p(self, actions_index):
        input = torch.LongTensor([0]).to(self.device)
        h_t, c_t = self.init_hidden()
        t = 0
        actions_p = []
        actions_log_p = []
        len_std_per_modality = [0 for i in range(self.num_modality)]

        # Automatic generation of modality-specific preprocessing schemes.
        for modality in range(self.num_modality):
            item = None
            while (item != 'stop' and len_std_per_modality[modality] < self.max_size):
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.std_decoder)
                action_index = actions_index[t].unsqueeze(0)
                t += 1
                p = F.softmax(logits, dim=-1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0, action_index]
                actions_p.append(p)
                actions_log_p.append(log_p)

                input = action_index
                item = self.lib.nodes_std[action_index]
                len_std_per_modality[modality] += 1

        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)

        return actions_p, actions_log_p