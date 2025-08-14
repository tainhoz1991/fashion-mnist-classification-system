import torch
import torch.nn.functional as F


def nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the negative log likelihood loss
    :param output: the result from the model after being applied log_softmax, shape (N, C)
    :param target: the index of column (C) of each rows in N rows from that the value is used to calculate NLLLoss, shape (N)
    """
    return F.nll_loss(output, target)
