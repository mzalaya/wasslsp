




def np_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(np_cdf_loss(tensor_a,tensor_b,p=1))

def np_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*np_cdf_loss(tensor_a,tensor_b,p=2))

def np_cdf_loss(tensor_a, tensor_b, p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (np.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (np.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = np.cumsum(tensor_a, dim=-1)
    cdf_tensor_b = np.cumsum(tensor_b, dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = np.sum(np.abs((cdf_tensor_a - cdf_tensor_b)), dim=-1)
    elif p == 2:
        cdf_distance = np.sqrt(np.sum(np.pow((cdf_tensor_a - cdf_tensor_b), 2), dim=-1))
    else:
        cdf_distance = np.pow(np.sum(np.pow(np.abs(cdf_tensor_a - cdf_tensor_b), p), dim=-1), 1 / p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss

def np_validate_distibution(tensor_a,tensor_b):
    # Zero sized dimension is not supported by np, we suppose there is no empty inputs
    # Weights should be non-negetive, and with a positive and finite sum
    # We suppose all conditions will be corrected by network training
    # We only check the match of the size here
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Input weight tensors must be of the same size")