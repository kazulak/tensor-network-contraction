import quimb.tensor as qtn

def build_random_regular(n, reg=3, D=8, seed=42):
    """Builds a random reg-regular graph tensor network.
    
    Args:
        n (int): Number of tensors (must be even).
        reg (int): Degree of each node (usually 3).
        D (int): Bond dimension.
        seed (int): Random seed.
        
    Returns:
        qtn.TensorNetworkGen: The tensor network.
    """
    if n % 2 != 0:
        raise ValueError(f"Number of tensors n must be even for regular graphs, got {n}")
    return qtn.TN_rand_reg(n=n, reg=reg, D=D, seed=seed)

def build_2d_grid_norm(L, D=5, seed=42):
    """Builds a closed 2D grid tensor network by contracting a 2D grid random network with its conjugate.
    
    Args:
        L (int): Grid width and height (L x L grid).
        D (int): Bond dimension.
        seed (int): Random seed.
        
    Returns:
        qtn.TensorNetworkGen: The closed tensor network representing the norm.
    """
    tn_base = qtn.TN2D_rand(L, L, D=D, seed=seed)
    return tn_base.H & tn_base
