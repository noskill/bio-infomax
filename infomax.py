import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


def shift1(Y):
    # shift by 1
    select = numpy.arange(1, len(Y) + 1)
    select[-1] = 0
    shifted = Y[select]
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
        # drive real to 1
        encoder_loss = (- real).mean()
        # discriminator
        disc_loss = (fake - real).mean()
        return dict(encoder_loss=encoder_loss,
                    discriminator_loss=disc_loss)


class GlobalDiscriminatorFull(nn.Module):
    # For the global mutual information objective, we first encode 
    # the input into a feature map, Cψ (x), which in this case is the output of the last convolutional layer. 
    # We then encode this representation further using linear layers as detailed above to get Eψ (x). Cψ (x) is then flattened, then concatenated with Eψ (x). We then pass this to a fully-connected network with two 512-unit hidden layers (see Table 6).
    def __init__(self, input_size):
        super().__init__()
        # like in table 6
        self.relu = nn.ReLU()
        self.layer0 = nn.Linear(input_size, 512)
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 1)

    def _forward(self, M, Y):
        # flatten M and concatenate, then pass through layers
        m1 = M.flatten(start_dim=1)
        x = torch.cat([m1, Y], dim=1)
        x = self.relu(self.layer0(x))
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        # 1 = real 0 = fake
        return torch.sigmoid(x)

    def forward(self, M, Y):
        real = self._forward(M, Y)
        shifted = shift1(Y)
        fake = self._forward(M, shifted)
        # just optimize for real samples 
        loss_encoder = (- real).mean()
        # discriminator should recognize both
        # minimum is when fake = 0 and real = 1
        loss_disc = (fake - real).mean()
        return dict(encoder_loss=loss_encoder,
                    discriminator_loss=loss_disc)


class InfoMax(nn.Module):
    def __init__(self, feature_network, 
                      aggregator_network, *losses):
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
        for loss in self.losses:
            l = loss(M, Y)
            for k, v in l.items():
                if k in result:
                    result[k] += v
                else:
                    result[k] = v
        return result 

