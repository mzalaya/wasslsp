# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>
# License:

import torch

def uniform(z, device=None):
    z = torch.tensor(z).to(device)
    if torch.abs(z) <= 1.:
        return 1.
    else:
        return 0.

def rectangle(z, device=None):
    z = torch.tensor(z).to(device)
    if torch.abs(z) <= 1.:
        return 1/2
    else:
        return 0.

def triangle(z, device=None):
    z = torch.tensor(z).to(device)
    if torch.abs(z) <= 1.:
        return (1 - torch.abs(z))
    else:
        return 0.

def epanechnikov(z, device=None):
    z = torch.tensor(z).to(device)
    if torch.abs(z) <= 1.:
        return 3/4 * (1 - z**2)
    else:
        return 0.
def biweight(z, device=None):

    mask = torch.abs(z) <= 1.
    z_mask = (15/16 * (1 - z**2)**2) * mask
    return z_mask


    # z : BatchedTensor(lvl=1, bdim=0, value= tensor([0.], device='mps:0')
    # z = torch._C._functorch.get_unwrapped(z)
    # z = z.detach().cpu()
    # print(z)
    # if torch.abs(z) <= 1.:
        # z = 15/16 * (1 - z**2)**2
        # return z
    # else:
        # return 0.

def tricube(z, device=None):
    z = torch.tensor(z).to(device)
    if torch.abs(z) <= 1.:
        return (1 - torch.abs(z) ** 3)*3
    else:
        return 0.

def gaussian(z, device=None):
    # z : BatchedTensor(lvl=1, bdim=0, value= tensor([0.], device='mps:0')
    # z = torch._C._functorch.get_unwrapped(z)
    # print(type(z))
    # print(z)
    # z = torch.tensor(z).to(device)
    return 1./torch.sqrt(2 * torch.as_tensor(torch.pi)) * torch.exp(-z ** 2 / 2)

def silverman(z, device=None):
    # z = torch.tensor(z).to(device)
    return 1/2 * torch.exp(-torch.abs(z) / torch.sqrt(torch.tensor([2.])) ) * torch.sin(torch.abs(z) / torch.sqrt(torch.tensor([2.])) + torch.pi / 4)


class ECDFTorch(torch.nn.Module):
    def __init__(self, x, weights=None, side='right', device=None):
        super(ECDFTorch, self).__init__()

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





def torch_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a,tensor_b,p=1))

def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))

def torch_cdf_loss(tensor_a, tensor_b, p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a - cdf_tensor_b)), dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a - cdf_tensor_b), p), dim=-1), 1 / p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss

def torch_validate_distibution(tensor_a,tensor_b):
    # Zero sized dimension is not supported by torch, we suppose there is no empty inputs
    # Weights should be non-negetive, and with a positive and finite sum
    # We suppose all conditions will be corrected by network training
    # We only check the match of the size here
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Input weight tensors must be of the same size")