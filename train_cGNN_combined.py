import json
import time
import os
import sys
import torch
from MultiLayerHuman import MultiLayerHuman
from tqdm import tqdm
from torch_geometric import seed_everything
from torch_geometric.utils import dropout_edge


def drop_edges(inp_graph, p=0.5):
    pyg = inp_graph.clone()
    pyg["rna", "links", "rna"].edge_index = dropout_edge(
        inp_graph["rna", "links", "rna"].edge_index, p=p
    )[0]
    pyg["protein", "interacts", "protein"].edge_index = dropout_edge(
        inp_graph["protein", "interacts", "protein"].edge_index, p=p
    )[0]
    pyg["rna", "synth", "protein"].edge_index = dropout_edge(
        inp_graph["rna", "synth", "protein"].edge_index, p=p
    )[0]
    pyg["protein", "prod", "metabolite"].edge_index = dropout_edge(
        inp_graph["protein", "prod", "metabolite"].edge_index, p=p
    )[0]
    return pyg


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
    n_runs = kwargs["config"]["model"]["runs"]
    _timestamp = int(time.time())
    global_min_loss = sys.maxsize
    global_best_weight = None

    for i in range(n_runs):
        print("\tRun No. = ", i + 1)
        seed_everything(kwargs["config"]["model"]["seed"][i])
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
        losses_val = []
        losses_val_prot = []
        losses_val_metab = []

        losses = []
        losses_prot = []
        losses_metab = []
        lrs = []
        min_val_loss = sys.maxsize

        for _ in tqdm(range(kwargs["epochs"])):
            model.train()

            # Re-create noise of intermediate nodes
            pyg["protein"].x = torch.randn(
                pyg["protein"].x.shape[0], pyg["protein"].x.shape[1]
            )
            pyg["metabolite"].x = torch.randn(
                pyg["metabolite"].x.shape[0], pyg["metabolite"].x.shape[1]
            )

            if kwargs["config"]["model"]["drop_edges"] > 0:
                pyg_train = drop_edges(pyg, p=kwargs["config"]["model"]["drop_edges"])

            optim.zero_grad()
            out = model(pyg_train.x_dict, pyg_train.edge_index_dict)

            loss1 = loss_fn(
                out["metabolite"][:, pyg["metabolite"].train_mask],
                pyg["metabolite"].y[:, pyg["metabolite"].train_mask],
            )
            loss2 = loss_fn(
                out["protein"][:, pyg["protein"].train_mask],
                pyg["protein"].y[:, pyg["protein"].train_mask],
            )
            loss_comb = (
                kwargs["config"]["model"]["metab_loss"] * loss1
                + kwargs["config"]["model"]["prot_loss"] * loss2
            )

            loss_comb.backward()
            optim.step()

            model.eval()
            with torch.no_grad():
                out_val = model(pyg.x_dict, pyg.edge_index_dict)
                loss1_val = loss_fn(
                    out_val["metabolite"][:, pyg["metabolite"].test_mask],
                    pyg["metabolite"].y[:, pyg["metabolite"].test_mask],
                )
                loss2_val = loss_fn(
                    out_val["protein"][:, pyg["protein"].test_mask],
                    pyg["protein"].y[:, pyg["protein"].test_mask],
                )

                loss_comb_val = (
                    kwargs["config"]["model"]["metab_loss"] * loss1_val
                    + kwargs["config"]["model"]["prot_loss"] * loss2_val
                )
            scheduler.step(loss_comb_val)

            if loss_comb_val < min_val_loss:
                best_model_weight = model.state_dict()
                min_val_loss = loss_comb_val.item()

            if loss_comb_val < global_min_loss:
                global_best_weight = model.state_dict()
                global_min_loss = loss_comb_val.item()

            losses_metab.append(loss1.item())
            losses_prot.append(loss2.item())
            losses.append(loss_comb.item())

            losses_val_metab.append(loss1_val.item())
            losses_val_prot.append(loss2_val.item())
            losses_val.append(loss_comb_val.item())

            lrs.append(scheduler.get_last_lr()[0])

        parent_dir = os.path.dirname(os.path.realpath(__file__))
        abs_path = os.path.join(
            parent_dir,
            kwargs["save_dir"],
            "weights",
            "run_" + str(i + 1) + kwargs["weight_path"],
        )
        torch.save(best_model_weight, abs_path)
        log_ = {
            "loss_val": {
                "prot": losses_val_prot,
                "metab": losses_val_metab,
                "comb": losses_val,
            },
            "loss_train": {"prot": losses_prot, "metab": losses_metab, "comb": losses},
            "lrs": lrs,
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
            str(_timestamp) + "_run_" + str(i) + ".json",
        )
        with open(log_path, "w") as f:
            json.dump(log_, f)

        print(f"\tTraining Completed. Saved logs at {log_path}")
    abs_path = os.path.join(
        parent_dir, kwargs["save_dir"], "weights", kwargs["weight_path"]
    )
    print("Saving global weight at ", abs_path)
    torch.save(global_best_weight, abs_path)
