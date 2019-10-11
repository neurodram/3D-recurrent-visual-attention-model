import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

import numpy as np


class retina(object):
    """
    A retina that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. It encodes
    the region around `l` at a high-resolution but uses
    a progressively lower resolution for pixels further
    from `l`, resulting in a compressed representation
    of the original image `x`.

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l: a 2D Tensor of shape (B, 2). Contains normalized
      coordinates in the range [-1, 1].
    - g: size of the first square patch.
    - k: number of patches to extract in the glimpse.
    - s: scaling factor that controls the size of
      successive patches.

    Returns
    -------
    - phi: a 5D tensor of shape (B, k, g, g, C). The
      foveated glimpse of the image.
    """
    def __init__(self, g, k, s):
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l):
        """
        Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        """
        phi = []
        size = self.g

        # extract k patches of increasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool3d(phi[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        #phi = phi.view(phi.shape[0], -1)

        return phi

    def extract_patch(self, x, l, size):
        """
        Extract a single patch for each image in the
        minibatch `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 2).
        - size: a scalar defining the size of the extracted patch.

        Returns
        -------
        - patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W, D = x.shape
        self.is_3d = (len(x.shape)==5)
        # denormalize coords of patch center
        coords = self.denormalize(x.shape[2:],l)

        # compute top left corner of patch
        patch_x = coords[:, 0] - (size // 2)
        patch_y = coords[:, 1] - (size // 2)
        if self.is_3d:
            patch_z = coords[:,2] - (size//2)
        # loop through mini-batch and extract
        patch = []
        for i in range(B):
            im = x[i].unsqueeze(dim=0)
            T = im.shape[2:]

            # compute slice indices
            from_x, to_x = patch_x[i], patch_x[i] + size
            from_y, to_y = patch_y[i], patch_y[i] + size
            if self.is_3d:
                from_z, to_z = patch_z[i], patch_z[i] + size
            # cast to ints
            from_x, to_x = from_x.item(), to_x.item()
            from_y, to_y = from_y.item(), to_y.item()
            if self.is_3d:
                from_z, to_z = from_z.item(), to_z.item()
            if self.is_3d:
                # pad tensor in case exceeds
                if self.exceeds(from_x, to_x, from_y, to_y, T, from_z, to_z):
                    pad_dims = (
                        size//2+1, size//2+1,
                        size//2+1, size//2+1,
                        size//2+1, size//2+1,
                        0, 0,
                        0, 0,
                    )
                    im = F.pad(im, pad_dims, "constant", im[0,:,-2:,-2:,-2:].mean().item())

                    # add correction factor
                    from_x += (size//2+1)
                    to_x += (size//2+1)
                    from_y += (size//2+1)
                    to_y += (size//2+1)
                    from_z += (size//2+1)
                    to_z += (size//2+1)

                # and finally extract
                patch.append(im[:, :, from_x:to_x, from_y:to_y, from_z:to_z])
            else: 
                # pad tensor in case exceeds
                if self.exceeds(from_x, to_x, from_y, to_y, T):
                    pad_dims = (
                        size//2+1, size//2+1,
                        size//2+1, size//2+1,
                        0, 0,
                        0, 0,
                    )
                    im = F.pad(im, pad_dims, "constant", im[0,:,-2:,-2:].mean().item())

                    # add correction factor
                    from_x += (size//2+1)
                    to_x += (size//2+1)
                    from_y += (size//2+1)
                    to_y += (size//2+1)

                # and finally extract
                patch.append(im[:, :, from_x:to_x, from_y:to_y])

        # concatenate into a single tensor
        patch = torch.cat(patch)

        return patch

    def denormalize(self,dims,l):
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        #return (0.5 * ((coords + 1.0) * T)).long()
        coords = torch.zeros(l.shape[0],l.shape[1])
        for i in range(l.shape[1]):
            coords[:,i] = (0.5 * ((l[:,i] + 1.0) * dims[i])).long()

        return coords.long()

    def exceeds(self, from_x, to_x, from_y, to_y, T, from_z = None, to_z = None):
        """
        Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if self.is_3d:
            if ((from_x < 0) or (from_y < 0) or (from_z < 0) or (to_x > T[0]) or (to_y > T[1])
                or (to_z > T[2])):
                return True
            return False

        else:
            if ((from_x < 0) or (from_y < 0) or (to_x > T[0]) or (to_y > T[1])):
                return True
            return False



class glimpse_3d(nn.Module):
    """The model we use in the paper."""

    def __init__(self, h_g, h_l, g, k, s, c):
        super(glimpse_3d,self).__init__()
        self.k = k
        self.retina = retina(g, k, s)
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_4 = nn.Conv3d(32,64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.fc = nn.Linear(2*2048,h_g)
        if self.k > 1:
            self.fc_glimpse = nn.Linear(4*2048,h_g)
        D_in = 3
        self.fc2 = nn.Linear(D_in, h_l)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x, l_t_prev,display,axes,labels,dem):
        x = self.retina.foveate(x, l_t_prev)
        if display:
            #for i in range(8):
                #print(dem[i].item())
                #np.save('/home/dw19/Desktop/traj_images/sag/'+ 'dem_{}_'.format(dem[i].item()) + 'im_{}_label_{}.npy'.format(len(os.listdir('/home/dw19/Desktop/traj_images/sag/'))+1,labels[i].item()),
                #x[i,:,20,:,:].squeeze().cpu())
                #np.save('/home/dw19/Desktop/traj_images/tran/'+ 'dem_{}_'.format(dem[i].item()) + 'im_{}_label_{}.npy'.format(len(os.listdir('/home/dw19/Desktop/traj_images/tran/'))+1,labels[i].item()),
                #x[i,:,:,:,20].squeeze().cpu())
                #np.save('/home/dw19/Desktop/traj_images/cor/'+ 'dem_{}_'.format(dem[i].item()) + 'im_{}_label_{}.npy'.format(len(os.listdir('/home/dw19/Desktop/traj_images/cor/'))+1,labels[i].item()),
                #x[i,:,:,20,:].squeeze().cpu())
            axes[0].imshow(np.flipud(np.transpose((x[1,:,20,:,:].squeeze().cpu()))))
            axes[1].imshow(np.flipud(np.transpose((x[1,:,:,:,20].squeeze().cpu()))))
            axes[2].imshow(np.flipud(np.transpose((x[1,:,:,20,:].squeeze().cpu()))))
            plt.title(labels[1].item())
            plt.pause(2)

            plt.cla()
            #print(l_t_prev.shape)
        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        l_out = F.relu(self.fc2(l_t_prev))
        if self.k == 1:
            x = self.pool(F.relu(self.Conv_1_bn(self.Conv_1(x))))
            x = self.pool(F.relu(self.Conv_2_bn(self.Conv_2(x))))
            x = F.relu(self.Conv_3_bn(self.Conv_3(x)))
            x = (self.Conv_4_bn(self.Conv_4(x)))
            x = F.relu(self.fc(x.view(x.shape[0], -1)))
        else:
            for i in range(self.k):
                out = self.pool(F.relu(self.Conv_1_bn(self.Conv_1(x[:,i,:,:,:].unsqueeze(1)))))
                out = self.pool(F.relu(self.Conv_2_bn(self.Conv_2(out))))
                out = F.relu(self.Conv_3_bn(self.Conv_3(out)))
                out = (self.Conv_4_bn(self.Conv_4(out)))
                out = out.view(out.shape[0], -1)
                if i == 0:
                    temp = out
                else:
                    temp = torch.cat([temp,out],dim=1)
            x = F.relu(self.fc_glimpse(temp))
                
                
        what = x
        where = l_out
        # feed to fc layer
        g_t = F.relu(torch.mul(what,where))
        return g_t



class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, h_g, h_l, g, k, s, c):
        super(glimpse_network, self).__init__()
        self.retina = retina(g, k, s)

        # glimpse layer
        D_in = k*g*g*c
        self.fc1 = nn.Linear(D_in, h_g)

        # location layer
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_l)

        #self.fc3 = nn.Linear(h_g, h_g+h_l)
        #self.fc4 = nn.Linear(h_l, h_g+h_l)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))

        #what = self.fc3(phi_out)
        #where = self.fc4(l_out)
        what = phi_out
        where = l_out
        # feed to fc layer
        g_t = F.relu(torch.mul(what,where))

        return g_t

class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """
    def __init__(self, input_size, hidden_size):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_1 = nn.LSTM(input_size,hidden_size,1)
        self.lstm_2 = nn.LSTM(input_size,hidden_size,1)

    def forward(self, g_t, h_1_prev, c_1_prev, h_2_prev, c_2_prev):
        h_1, (_,c_1) = self.lstm_1(g_t, (h_1_prev, c_1_prev))
        h_2, (_,c_2) = self.lstm_2(h_1, (h_2_prev, c_2_prev))
        return h_1, c_1, h_2, c_2


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_1):
        a_t = F.log_softmax(self.fc(h_1.squeeze()), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """
    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        # compute mean

        mu = 0.5*F.tanh(self.fc(h_t.squeeze(0)))
        #mu = torch.clamp(mu,-.5,.5)
        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise
        # bound between [-1, 1]
        l_t = torch.clamp(l_t,-1,1)
        #l_t =F.tanh(l_t)
        l_t = l_t.detach()

        return mu, l_t


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t.detach()))
        return b_t

class context_network(nn.Module):
    """ 
    CNN which takes coarse glimpse to provide hints for where to look.
    The output is passed to the location network to provide l_0, and also 
    becomes the initial hidden layer for the second lstm.
    """
    def __init__(self,hidden_size):
        super(context_network, self).__init__()
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,32,3)
        self.conv3 = nn.Conv2d(32,32,3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(800,hidden_size)
        self.pool3 = nn.MaxPool2d(3)

    def forward(self, x):
        x = self.pool3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(self.fc(x.view(x.shape[0], -1)))
        
        return x.unsqueeze(0)

class context_network_clin(nn.Module):
    """ 
    CNN which takes coarse glimpse to provide hints for where to look.
    The output is passed to the location network to provide l_0, and also 
    becomes the initial hidden layer for the second lstm.
    """
    def __init__(self,hidden_size):
        super(context_network_clin, self).__init__()
        self.fc = nn.Linear(13,hidden_size)

    def forward(self, x):
        out = F.relu(self.fc(x)).unsqueeze(0)
        
        return out









