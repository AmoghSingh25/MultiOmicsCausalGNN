import pickle
import torch

def _save_file(path, var):
    with open(path, "wb") as file:
        pickle.dump(var, file)


def _read_file(path):
    with open(path, "rb") as file:
        var = pickle.load(file)
    return var


def _read_txt(path):
    with open(path, "r") as file:
        var = file.readlines()
    var = [x.strip() for x in var]
    return var

def z_score_norm(inp, axis=0):
    mean = torch.mean(inp, axis=axis)
    std = torch.std(inp, axis=axis)

    inp = (inp-mean)/ std
    return inp

def min_max_norm(inp, axis=0, eps=1e-8):
    arr_max = torch.max(inp, dim=axis).values
    arr_min = torch.min(inp, dim=axis).values

    inp = (inp - arr_min)/(arr_max - arr_min + eps)
    return inp