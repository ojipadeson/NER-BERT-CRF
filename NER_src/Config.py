import torch

cuda_yes = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_yes else "cpu")

max_seq_length = 180
