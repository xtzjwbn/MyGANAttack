import torch
import torch.nn as nn
import numpy as np


class AutoEncoder(nn.Module):
    """
        NN model used to translate data to latent vector
        Encoder : data -> latent vector
        Decoder : latent vector -> data
    """
    def __init__(self,input_dim = 100, code_dim = 8):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),   #256
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),    # 256 -> 128
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),     # 128 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64,code_dim),   # 64
            # nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 64),   # 64
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),       # 64->128
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),      # 128->256
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, input_dim),   # 256
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def transform(self, x):
        """
        :param x: data
        :return: latent vector
        """
        return self.encoder(x)

    def inverse_transform(self, r):
        """
        :param r: latent vector
        :return: data
        """
        return self.decoder(r)


class netClassificationMLP(nn.Module):
    """
    Classifier Model used to classificate raw data to 0-1
    """
    def __init__(self, inputDim, outputDim):
        super(netClassificationMLP, self).__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.model = nn.Sequential(
            nn.Flatten(),

            #using residual
            # Residual(self.inputDim),

            nn.Linear(self.inputDim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            Residual(32),
            nn.Linear(32, self.outputDim)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        return self.model(x)

class Residual(nn.Module):
    """Residual layer from resnet."""

    def __init__(self, i):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, i)
        self.bn = nn.BatchNorm1d(i)
        self.relu = nn.ReLU()

    def forward(self, input_):
        return self.relu(self.bn(self.fc(input_)) + input_)


class AdvGenerator(nn.Module):
    """
        Generator : noise | condition to new vector
    """
    def __init__(self, input_dim, output_dim):
        super(AdvGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.model(x)

class AdvDiscriminator(nn.Module):
    """
    Discriminator you know who
    """
    def __init__(self, dataDim):
        super(AdvDiscriminator, self).__init__()
        self.dataDim = dataDim
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dataDim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
