from math import inf
import numpy as np
import torch

def torch_softmax(x: torch.Tensor):
    return torch.softmax(x, dim=-1)

def naive_softmax(x:np.ndarray):
    softmax = []
    for row in x:
        d = np.sum(np.exp(row))
        for ele in row:
            softmax.append((np.exp(ele)/d).item())

    return softmax

def safe_softmax(x:np.ndarray):
    softmax = []
    for row in x:
        # iter 1: finding max
        max = np.max(row)
        # iter 2: calculating denominator
        d = np.sum(np.exp(row - max))
        # iter 3: softmax operation
        for ele in row:
            softmax.append((np.exp(ele - max)/d).item())
    return softmax

def online_softmax(x:np.ndarray):
    softmax = []
    for row in x:
        # iter 1: fused pass - running max and d
        max = -inf
        d = 0
        for i, ele in enumerate(row):
            if i > 0:
                old_max = max
                max = np.max([ele, old_max])
                d = d * np.exp(old_max - max) + np.exp(ele - max)
            else:
                max = ele
                d = 1

        # iter 2: softmax operation
        for ele in row:
            softmax.append((np.exp(ele - max)/d).item())
    return softmax


matrix = np.random.randint(0, 1, (5,5), dtype=np.int8)
print(torch_softmax(torch.Tensor(matrix)))
print(naive_softmax(matrix))
print(safe_softmax(matrix))
print(online_softmax(matrix))
