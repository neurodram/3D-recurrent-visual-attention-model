import math

import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import baseline_network, context_network, context_network_clin
from modules import glimpse_network, core_network, glimpse_3d
from modules import action_network, location_network, retina


class RecurrentAttention(nn.Module):
    """
    A 3D recurrent visual attention model for interpretable neuroimaging 
    classification, as presented in https://arxiv.org/abs/1910.04721. 
    """
    def __init__(self,
                 g,
                 h_g,
                 h_l,
                 std,
                 hidden_size,
                 num_classes):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - h_g: hidden layer size of the fc layer for 'what' representation
        - h_l: hidden layer size of the fc layer for 'where' representation
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the LSTM
        - num_classes: number of classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()
        self.std = std
        self.ret = retina(g,k,s)

        self.sensor = glimpse_3d(h_g, h_l, g, k, s, c)
        self.rnn = core_network(hidden_size, hidden_size)
        self.locator = location_network(hidden_size, 3, std)
        self.classifier = action_network(hidden_size, num_classes)
        self.baseliner = baseline_network(hidden_size, 1)
        self.context = context_network_clin(hidden_size)

    def forward(self, x, l_t, h_1, c_1, h_2, c_2, first = False, last=False):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 5D Tensor of shape (B, C, H, W, D). The minibatch
          of images.
        - l_t_prev: a 3D tensor of shape (B, 3). The location vector
          containing the glimpse coordinates [x, y,z] for the previous
          timestep t-1.
        - h_1_prev, c_1_prev: a 2D tensor of shape (B, hidden_size). The 
          lower LSTM hidden state vector for the previous timestep t-1.
        - h_2_prev, c_2_prev: a 2D tensor of shape (B, hidden_size). The 
          upper LSTM hidden state vector for the previous timestep t-1.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline b_t for the
          current timestep t. 
          
        Returns
        -------
        - h_1_t, c_1_t, h_2_t, c_2_t: hidden LSTM states for current step
        - mu: a 3D tensor of shape (B, 3). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 3D tensor of shape (B, 3). The location vector
          containing the glimpse coordinates [x, y,z] for the
          current timestep t.
        - b_t: a vector of length (B,). The baseline for the
          current time step t.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        if first: #if t=0, return get first location to attend to  using the context
            h_0 = self.context(x)
            mu_0, l_0 = self.locator(h_0,)
            return h_0, l_0
        g_t = self.sensor(x, l_t,display,axes,labels,dem)
        h_1, c_1, h_2, c_2 = self.rnn(g_t.unsqueeze(0), h_1, c_1, h_2, c_2)
        mu, l_t = self.locator(h_2)
        b_t = self.baseliner(h_2).squeeze()


        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1) #policy probabilities for REINFORCE training
        
        if last:
            log_probas = self.classifier(h_1) # perform classification and get class probabilities
            return h_1, c_1, h_2, c_2, l_t, b_t, log_probas, log_pi

        return h_1, c_1, h_2, c_2, l_t, b_t, log_pi
