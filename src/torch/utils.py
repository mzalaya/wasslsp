# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>
# License:

import torch
import math

def uniform(z, device=None):
    mask = torch.abs(z) <= 1.
    return mask * 1.


def rectangle(z, device=None):

    mask = torch.abs(z) <= 1.
    z_mask = 1/2 * mask
    return z_mask


def triangle(z, device=None):
    mask = torch.abs(z) <= 1.
    z_mask = (1 - torch.abs(z)) * mask

    return z_mask


def epanechnikov(z, device=None):
    mask = torch.abs(z) <= 1.
    z_mask = 3/4 * (1 - z**2) * mask
    return z_mask
def biweight(z, device=None):

    mask = torch.abs(z) <= 1.
    z_mask = (15/16 * (1 - z**2)**2) * mask
    return z_mask


def tricube(z, device=None):
    z = torch.tensor(z).to(device)

    mask = torch.abs(z) <= 1.
    z_mask = (1 - torch.abs(z) ** 3)**3 * mask
    return z_mask


def gaussian(z, device=None):
    return 1./torch.sqrt(2 * torch.as_tensor(torch.pi)) * torch.exp(-z ** 2 / 2)

def silverman(z, device=None):
    sqrt_2 = math.sqrt(2)
    return 1/2 * torch.exp(-torch.abs(z) / sqrt_2 ) * torch.sin(torch.abs(z) / sqrt_2 + torch.pi / 4)


class ECDFtorch(torch.nn.Module):
    def __init__(self, x, weights=None, side='right', device=None):
        super(ECDFtorch, self).__init__()

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side
        self.device = device

        if len(x.shape) != 1:
            msg = 'x must be 1-dimensional'
            raise ValueError(msg)

        nobs = len(x)
        if weights is not None:
            assert len(weights) == nobs
            sweights = torch.sum(weights)
            assert sweights > 0.
            sorted = torch.argsort(x).int()
            x = x[sorted]
            y = torch.cumsum(weights[sorted], dim=0)
            y = y / sweights
            self.x = torch.cat((torch.tensor([-torch.inf], device=x.device), x))
            self.y = torch.cat((torch.tensor([0], device=y.device), y))
            self.n = self.x.shape[0]

        else:
            x = torch.sort(x)[0]
            y = torch.linspace(1. / nobs, 1, nobs, device=x.device)
            self.x = torch.cat((torch.tensor([-torch.inf], device=x.device), x))
            self.y = torch.cat((torch.tensor([0], device=y.device), y))
            self.n = self.x.shape[0]

    def forward(self, time):
        tind = torch.searchsorted(self.x, time, side=self.side) - 1
        return self.y[tind].to(self.device)
