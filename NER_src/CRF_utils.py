import numpy as np
import torch


def log_sum_exp_1vec(vec):
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(log_M, axis=-1):
    return torch.max(log_M, axis)[0] + torch.log(torch.exp(log_M - torch.max(log_M, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_Tensor, axis=-1):
    return torch.max(log_Tensor, axis)[0] + torch.log(
        torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))
