import json
import time
import os
import sys
import torch
from MultiLayerHuman import MultiLayerHuman
from tqdm import tqdm
from torch_geometric import seed_everything
from omicsGraphDataset import OmicGraphDataset
from torch_geometric.loader import DataLoader


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
    cfg_device = kwargs["config"]["model"]["device"]
    device = torch.device(cfg_device)
    n_runs = kwargs["config"]["model"]["runs"]
    _timestamp = int(time.time())
    global_min_loss = sys.maxsize
    global_best_weight = None

    for i in range(n_runs):
        print("\tRun No. = ", i + 1)

        seed_everything(kwargs["config"]["model"]["seed"][i])
        train_dataset = OmicGraphDataset(pyg, device=cfg_device)
        test_dataset = OmicGraphDataset(pyg, training=False, device=cfg_device)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=8)

        model = MultiLayerHuman(inp_dim=1).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.01, patience=5, min_lr=1e-12
        )
        loss_fn = torch.nn.MSELoss()

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
            loss_p, loss_m, loss_p_test, loss_m_test = 0, 0, 0, 0
            loss_comb_train, loss_comb_test = 0, 0

            # Loop over train samples
            for sample_x in train_loader:
                optim.zero_grad()
                out = model(sample_x.x_dict, sample_x.edge_index_dict)

                loss1 = loss_fn(
                    out["metabolite"],
                    sample_x["metabolite"].y,
                )
                loss2 = loss_fn(
                    out["protein"],
                    sample_x["protein"].y,
                )

                loss_comb = (
                    kwargs["config"]["model"]["metab_loss"] * loss1
                    + kwargs["config"]["model"]["prot_loss"] * loss2
                )

                loss_m = loss_m + loss1
                loss_p = loss_p + loss2
                loss_comb_train = loss_comb_train + loss_comb

                loss_comb.backward()
                optim.step()
            scheduler.step(loss_comb)

            # Loop over test samples
            with torch.no_grad():
                for sample_x in test_loader:
                    out = model(sample_x.x_dict, sample_x.edge_index_dict)

                    loss1_test = loss_fn(
                        out["metabolite"],
                        sample_x["metabolite"].y,
                    )
                    loss2_test = loss_fn(
                        out["protein"],
                        sample_x["protein"].y,
                    )
                    loss_m_test = loss_m_test + loss1_test
                    loss_p_test = loss_p_test + loss2_test

                    loss_comb_test_i = (
                        kwargs["config"]["model"]["metab_loss"] * loss1_test
                        + kwargs["config"]["model"]["prot_loss"] * loss2_test
                    )
                    loss_comb_test = loss_comb_test + loss_comb_test_i

            loss_m = loss_m / train_dataset.len()
            loss_p = loss_p / train_dataset.len()
            loss_comb_train = loss_comb_train / train_dataset.len()

            loss_m_test = loss_m_test / test_dataset.len()
            loss_p_test = loss_p_test / test_dataset.len()
            loss_comb_test = loss_comb_test / test_dataset.len()

            if loss_comb_test < min_val_loss:
                best_model_weight = model.state_dict()
                min_val_loss = loss_comb_test.item()

            if loss_comb_test < global_min_loss:
                global_best_weight = model.state_dict()
                global_min_loss = loss_comb_test.item()

            losses_metab.append(loss_m.item())
            losses_prot.append(loss_p.item())
            losses.append(loss_comb_train.item())

            losses_val_metab.append(loss_m_test.item())
            losses_val_prot.append(loss_p_test.item())
            losses_val.append(loss_comb_test.item())

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
