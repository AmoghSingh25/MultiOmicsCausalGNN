import os
import numpy as np
import torch
from MultiLayerHuman import MultiLayerHuman
from utils import _save_file, _read_file
from torch_geometric import seed_everything


def eval_model(pyg, weight_path, seed, device):
    seed_everything(seed)

    model = MultiLayerHuman(inp_dim=pyg["protein"].x.shape[1]).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        out = model(pyg.x_dict, pyg.edge_index_dict)

    test_mask = pyg["metabolite"].test_mask

    norm_metab_output = out["metabolite"][:, test_mask]
    norm_prot_output = out["protein"][:, test_mask]

    norm_metab_target = pyg["metabolite"].y[:, test_mask]
    norm_prot_target = pyg["protein"].y[:, test_mask]

    metab_val_loss = loss_fn(norm_metab_output, norm_metab_target)
    prot_val_loss = loss_fn(norm_prot_output, norm_prot_target)
    return prot_val_loss.item(), metab_val_loss.item()


def save_stats(network_name, pyg, path, weight_path, seed, device):
    seed_everything(seed)

    model = MultiLayerHuman(inp_dim=pyg["protein"].x.shape[1]).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    with torch.no_grad():
        out = model(pyg.x_dict, pyg.edge_index_dict)

    test_mask = pyg["metabolite"].test_mask

    norm_metab_output = out["metabolite"][:, test_mask]
    norm_prot_output = out["protein"][:, test_mask]

    norm_metab_target = pyg["metabolite"].y[:, test_mask]
    norm_prot_target = pyg["protein"].y[:, test_mask]

    metric = _read_file(path)
    metric[network_name] = {}
    # metric[network_name]["predicted_prot"] = torch.std(norm_prot_output, dim=0)
    metric[network_name]["pred_prot"] = torch.std(norm_prot_output, dim=1)
    metric[network_name]["pred_metab"] = torch.std(norm_metab_output, dim=1)

    metric[network_name]["target_prot"] = torch.std(norm_prot_target, dim=1)
    metric[network_name]["target_metab"] = torch.std(norm_metab_target, dim=1)
    return metric


def get_weight_scores(pyg, output_path, metab_ratio, prot_ratio, seeds, device):
    weight_dir = os.path.join(output_path, "weights")
    weight_files = os.listdir(weight_dir)
    weight_files.sort()

    losses = []
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "output", "comp_metrics.pkl"
    )
    if not os.path.exists(output_dir):
        _save_file(output_dir, {})

    for i in range(len(weight_files[:-1])):
        if type(pyg) is list:
            pyg_i = pyg[i]
        else:
            pyg_i = pyg
        pyg_i.to(device)
        weight_dir_i = os.path.join(weight_dir, weight_files[i])
        prot_loss, metab_loss = eval_model(pyg_i, weight_dir_i, seeds[i], device=device)
        losses.append(
            (prot_loss, metab_loss, metab_loss * metab_ratio + prot_loss * prot_ratio)
        )

    losses = np.array(losses)

    print(output_path)
    print("Min - ", np.min(losses, axis=0))
    print("Mean - ", np.mean(losses, axis=0))
    print("Std - ", np.std(losses, axis=0))
