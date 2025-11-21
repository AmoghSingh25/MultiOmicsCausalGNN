## TO DO
# Parameters for LR, Loss function, combined loss function or single omic loss function
#
import json
import time
import os
import sys
import copy
import torch
from MultiLayerHuman import MultiLayerHuman
from tqdm import tqdm


def _trainGNN(pyg, **kwargs):
    """
    # Omic data containing the number of samples and the features of each sample
    # rna_data, prot_data, metab_data - (Number of samples, Number of features)
    # Features could be RNAs, proteins, metabolites
    # Ensure number of samples across omics is same


    # Edges configuration - Contains the list of edges between the nodes of the omics features
    # rr_edges - RNA-RNA links
    # pp_edges - Protein-Protein links
    # pr_edges - Protein-RNA links
    # pm_edges - Protein-Metabolite links
    # rr_edges, pp_edges, pr_edges, pm_edges - (Number of edges, 2)
    # Each pair contains the index of the feature from the first and second omic data respectively
    # Example, pm_edges - (Index of Protein feature, Index of Metabolite feature)
    """

    device = torch.device("cpu")

    model = MultiLayerHuman(inp_dim=pyg["protein"].x.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.01, patience=5, min_lr=1e-12
    )
    loss_fn = torch.nn.MSELoss()

    pyg = pyg.to(device)
    model = model.to(device)

    print("\tModel validation result = ", pyg.validate())

    best_model_weight = None
    model.train()
    losses = []
    losses_val = []
    lrs = []
    min_val_loss = sys.maxsize
    for _ in tqdm(range(kwargs["epochs"])):
        optim.zero_grad()
        out = model(pyg.x_dict, pyg.edge_index_dict)
        loss1 = loss_fn(
            out["metabolite"][:, pyg["metabolite"].train_mask],
            pyg["metabolite"].y[:, pyg["metabolite"].train_mask],
        )
        loss_comb = loss1
        loss_comb.backward()
        optim.step()
        scheduler.step(loss_comb)

        with torch.no_grad():
            loss_1 = loss_fn(
                out["metabolite"][:, pyg["metabolite"].test_mask],
                pyg["metabolite"].y[:, pyg["metabolite"].test_mask],
            )

        if loss_comb < min_val_loss:
            best_model_weight = copy.deepcopy(model.state_dict())
            min_val_loss = loss_comb
        losses.append(loss1.item())
        lrs.append(scheduler.get_last_lr()[0])
        losses_val.append(loss_1.item())

    parent_dir = os.path.dirname(os.path.realpath(__file__))
    abs_path = os.path.join(
        parent_dir, kwargs["save_dir"], "weights", kwargs["weight_path"]
    )
    torch.save(best_model_weight, abs_path)
    _timestamp = int(time.time())
    log_ = {
        "loss_val": losses,
        "loss_train": losses_val,
        "timestamp": _timestamp,
        "config": kwargs["config"],
        "name": time.time()
        if kwargs["config"]["name"] is None
        else kwargs["config"]["name"],
    }
    log_path = os.path.join(
        parent_dir,
        kwargs["config"]["output"]["save_dir"],
        "logs",
        str(_timestamp) + ".json",
    )
    with open(log_path, "w") as f:
        json.dump(log_, f)

    print(f"\tTraining Completed. Saved logs at {log_path}")
