import math

import torch
import torch.nn as nn

from torch.distributions import Normal

from modules import baseline_network, context_network, context_network_clin
from modules import glimpse_network, core_network, glimpse_3d
from modules import action_network, location_network, retina


class RecurrentAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """
    def __init__(self,
                 g,
                 k,
                 s,
                 c,
                 h_g,
                 h_l,
                 std,
                 hidden_size,
                 num_classes, load_params=False):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - c: number of channels in each image.
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the rnn.
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


        if load_params:
            self.sensor.load_state_dict(torch.load('/home/dw19/Desktop/glimpse_long.pt'))
            self.rnn.load_state_dict(torch.load('/home/dw19/Desktop/rnn_long.pt')) 
            self.locator.load_state_dict(torch.load('/home/dw19/Desktop/locator_long.pt')) 
            self.classifier.load_state_dict(torch.load('/home/dw19/Desktop/classifier_long.pt'))
            self.baseliner.load_state_dict(torch.load('/home/dw19/Desktop/baseliner_long.pt'))
            self.context.load_state_dict(torch.load('/home/dw19/Desktop/context_long.pt'))


    def forward(self, x, l_t, h_1, c_1, h_2, c_2, display=False, axes = None, first = False, last=False,labels = None, dem = None):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a vector of length (B,). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        if first:
            h_0 = self.context(x)
            mu_0, l_0 = self.locator(h_0,)
            return h_0, l_0
        g_t = self.sensor(x, l_t,display,axes,labels,dem)
        h_1, c_1, h_2, c_2 = self.rnn(g_t.unsqueeze(0), h_1, c_1, h_2, c_2)
        mu, l_t = self.locator(h_2)
        b_t = self.baseliner(h_2).squeeze()

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = Normal(mu, self.std).log_prob(l_t)
        log_pi = torch.sum(log_pi, dim=1)
        
        if last:
            log_probas = self.classifier(h_1)
            return h_1, c_1, h_2, c_2, l_t, b_t, log_probas, log_pi

        return h_1, c_1, h_2, c_2, l_t, b_t, log_pi
