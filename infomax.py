import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import init_weights


def shift1(Y):
    # shift by 1
    select = numpy.arange(1, len(Y) + 1)
    select[-1] = 0
    shifted = Y[torch.as_tensor(select, dtype=int)]
    return shifted


class LocalDiscriminator(nn.Module):
    #  Concat-and-convolve architecture
    def __init__(self, conv_in, y_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.layer0 = nn.Linear(y_size, 2048)
        self.layer1 = nn.Linear(y_size, 2048)
        # encoder for matrix M
        self.conv1 = nn.Conv2d(conv_in, 2048, 1, 1)
        self.conv2 = nn.Conv2d(conv_in, 2048, 1, 1)
        init_weights(self)

    def _forward(self, M, Y):
        # implementation follows the one from article
        x = self.relu(self.conv1(M))
        linear = self.conv2(M)
        local_emb = F.normalize(x + linear)

        g = self.relu(self.layer0(Y))
        linear = self.layer1(Y)
        global_emb = F.normalize(g + linear)

        # we have scores from -1 to 1
        prod = (global_emb.unsqueeze(1) @ local_emb.flatten(2))
        # move to [0, 1] range
        prod = (prod + 1) / 2
        return prod

    def forward(self, M, Y):
        real = self._forward(M, Y)
        shifted = shift1(Y)
        fake = self._forward(M, shifted)
        eps = 0.000001
        # drive real to 1
        encoder_loss = - torch.log(real + eps).mean()

        # discriminator
        # drive fake to zero and real to 1
        disc_loss = - torch.log(1 - fake + eps).mean() + encoder_loss
        return dict(local_encoder_loss=encoder_loss,
                    local_real=real.mean(),
                    local_fake=fake.mean(),
                    local_discriminator_loss=disc_loss)


class PriorDiscriminator(nn.Module):
    def __init__(self, y_size, bn=True):
        super().__init__()
        # small fully connected network
        mid = 2048
        self.layer0 = nn.Linear(y_size, mid)
        self.layer1 = nn.Linear(mid, mid)
        if bn:
            self.bn0 = nn.BatchNorm1d(mid)
            self.bn1 = nn.BatchNorm1d(mid)
        else:
            self.bn0 = lambda x: x
            self.bn1 = lambda x: x
        self.layer2 = nn.Linear(mid, 1)
        self.relu = nn.LeakyReLU()
        init_weights(self)

    def _forward(self, x):
        x = self.relu(self.bn0(self.layer0(x)))
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.layer2(x)
        return torch.sigmoid(x)

    def forward(self, M, Y):
        empirical = self._forward(Y)
        # now sample prior distribution
        prior_x = torch.normal(torch.zeros_like(Y), torch.ones_like(Y))
        prior = self._forward(prior_x)
        eps = 0.000001
        # that's for encoder only!
        # maximize empirical
        encoder_loss = - torch.log(empirical + eps).mean()
        # discriptor should classify empirical as fake
        # so minimize empirical and maximize prior
        disc_loss = - torch.log(1 - empirical + eps).mean() - torch.log(prior + eps).mean()
        return dict(prior_encoder_loss=encoder_loss,
                    prior_emp=empirical.mean(),
                    prior_fake=prior.mean(),
                    prior_discriminator_loss=disc_loss)


class LocalDiscriminatorConv(nn.Module):
    # We used a1×1convnet with two512-unit hidden layers as discriminator
    def __init__(self, conv_in, y_size):
        super().__init__()
        self.activation = nn.ReLU()
        mid = 512
        self.layer0 = nn.Conv2d(conv_in + y_size, mid, 1, 1)
        self.conv2 = nn.Conv2d(mid, mid, 1, 1)
        self.conv3 = nn.Conv2d(mid, 1, 1, 1)
        init_weights(self)

    def _forward(self, M, Y):
        Y = torch.repeat_interleave(Y.unsqueeze(2), M.shape[2], dim=2)
        Y = torch.repeat_interleave(Y.unsqueeze(3), M.shape[3], dim=3)
        x = torch.cat([M, Y], dim=1)
        x = self.activation(self.layer0(x))
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

    def forward(self, M, Y):
        real = self._forward(M, Y)
        shifted = shift1(Y)
        fake = self._forward(M, shifted)
        eps = 0.000001
        encoder_loss = - torch.log(real + eps).mean()
        disc_loss = - torch.log(1 - fake + eps).mean() + encoder_loss
        return dict(local_encoder_loss=encoder_loss,
                    local_real=real.mean(),
                    local_fake=fake.mean(),
                    local_discriminator_loss=disc_loss)


class GlobalDiscriminatorFull(nn.Module):
    # For the global mutual information objective, we first encode
    # the input into a feature map, Cψ (x), which in this case is the output of the last convolutional layer.
    # We then encode this representation further using linear layers as detailed above to get Eψ (x). Cψ (x) is then flattened, then concatenated with Eψ (x). We then pass this to a fully-connected network with two 512-unit hidden layers (see Table 6).
    def __init__(self, input_size):
        super().__init__()
        # like in table 6
        self.input_size = input_size
        self.relu = nn.ReLU()
        self.layer0 = nn.Linear(input_size, 512)
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 1)
        init_weights(self)

    def _forward(self, M, Y):
        # flatten M and concatenate, then pass through layers
        m1 = M.flatten(start_dim=1)
        x = torch.cat([m1, Y], dim=1)

        # check
        assert x.shape[1] == self.input_size
        x = self.relu(self.layer0(x))
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        # 1 = real 0 = fake
        return torch.sigmoid(x)

    def forward(self, M, Y):
        real = self._forward(M, Y)
        shifted = shift1(Y)
        fake = self._forward(M, shifted)
        eps = 0.0000001
        # just optimize for real samples
        # log(1) equals 0
        # it will drive reals to 1
        loss_encoder = - torch.log(real + eps).mean()
        # minimum is when fake = 0 and real = 1
        # log(1 - 0) equals 0
        # so it will drive fake to 0
        loss_disc = - torch.log(1 - fake + eps).mean() + loss_encoder

        return dict(global_encoder_loss=loss_encoder,
                    global_fake=fake.mean(),
                    global_real=real.mean(),
                    global_discriminator_loss=loss_disc)


class InfoMax(nn.Module):
    def __init__(self, feature_network,
                      aggregator_network, losses):
        super().__init__()
        self.feature_network = feature_network
        self.aggregator_network = aggregator_network
        self.losses = losses

    def forward(self, batch):
        """
        Deep InfoMax with different losses
        """
        # compute M (d*m*m) feature map for each image in the batch
        # compute global features Y
        # compute global loss, local loss and other losses

        M = self.feature_network(batch)
        Y = self.aggregator_network(M)
        result = dict()
        for loss in self.losses.values():
            l = loss(M, Y)
            for k, v in l.items():
                if k in result:
                    result[k] += v
                else:
                    result[k] = v
        return result, M, Y

