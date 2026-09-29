import torch

torch.set_num_threads(max(1, torch.get_num_threads()))
