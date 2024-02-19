import torch


def torch_uniform(size: tuple,
                  lower: float,
                  upper: float
                  ) -> torch.Tensor:
    """
    Returns a torch tensor with the given shape ``size`` that contains randomly uniformly 
    distributed values in the range from ``lower`` to ``upper``.
    
    :returns: Torch Tensor
    """
    return (lower-upper) * torch.rand(size=size) + upper 


def torch_gauss(size: tuple,
                mean: float,
                std: float,
                ) -> torch.Tensor:
    """
    Returns a torch tensor with the given shape ``size`` whose values are randomly generated 
    from a normal (gauss) distribution with ``mean`` and standard deviation ``std``
    
    :returns: Torch Tensor
    """
    tens_mean = torch.full(size=size, fill_value=mean, dtype=torch.float32)
    tens_std = torch.full(size=size, fill_value=std, dtype=torch.float32)
    return torch.normal(tens_mean, tens_std)