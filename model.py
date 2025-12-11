import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


class simdatset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y, index


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.name = 'ae'
        self.state = 'train' # or 'test'
        self.inputdim = input_dim # gene数
        self.outputdim = output_dim # cellType数

        # original demo
        # self.encoder = nn.Sequential(nn.Dropout(), # 0.2
        #                              nn.Linear(self.inputdim, 512),
        #                              nn.CELU(),
        #
        #
        #                              nn.Dropout(),
        #                              nn.Linear(512, 256),
        #                              nn.CELU(),
        #
        #
        #                              nn.Dropout(),
        #                              nn.Linear(256, 128),
        #                              nn.CELU(),
        #
        #                              nn.Dropout(),
        #                              nn.Linear(128, 64),
        #                              nn.CELU(),
        #
        #                              nn.Linear(64, output_dim),
        #                              )

        # Relu no Dropout demo (my ideal model)
        self.encoder = nn.Sequential(

                                     nn.Linear(self.inputdim, 512),
                                     nn.CELU(),

                                     nn.Dropout(),
                                     nn.Linear(512, 256),
                                     nn.CELU(),

                                     nn.Linear(256, 128),
                                     nn.CELU(),

                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),

                                     nn.Linear(64, output_dim),
                                     nn.ReLU(),
        )

        self.decoder = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.inputdim, bias=False))


    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self,x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x/x_sum
    
    def sigmatrix(self):
        weights = [layer.weight.T for layer in self.decoder]
        w = weights[0]
        for wi in weights[1:]:
            w = torch.mm(w, wi)
        return F.relu(w)

    def forward(self, x):
        z = self.encode(x)
        sigmatrix =  self.sigmatrix()
        x_recon = torch.mm(z, sigmatrix)
        return x_recon, z, sigmatrix

def reproducibility(seed=9):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True