import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import polars as pl
    from torch_geometric.nn import HeteroConv, HeteroDictLinear, SAGEConv, Linear
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric import EdgeIndex
    from torch_geometric.loader import DataLoader
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from torch.nn import Linear as LinearNN
    from sklearn.metrics import accuracy_score
    import pickle

    import sys
    from pathlib import Path

    PROJECT_ROOT = Path.cwd()  # or Path(__file__).resolve().parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from TCGA_LUAD.graphDataset import OmicGraphDataset
    return (
        DataLoader,
        EdgeIndex,
        HeteroConv,
        HeteroData,
        HeteroDictLinear,
        Linear,
        LinearNN,
        OmicGraphDataset,
        SAGEConv,
        accuracy_score,
        np,
        pickle,
        pl,
        plt,
        torch,
        tqdm,
    )


@app.function
def frmt_id(x):
    return x.replace("-", ".")[:-3]


@app.cell
def _(torch):
    device = torch.device("mps")
    return (device,)


@app.cell
def _(device, pl, torch):
    rna_data = pl.read_csv("TCGA_LUAD/Data/frmt_RNASeq.csv")
    mirna_data = pl.read_csv("TCGA_LUAD/Data/frmt_miRNA.csv")
    meth_data = pl.read_csv(
        "TCGA_LUAD/Data/frmt_Methylation.csv", null_values=["NA"]
    )
    labels = pl.read_csv("TCGA_LUAD/Data/luad_subtypes.tsv", separator="\t")

    sample_ids, y = (
        [frmt_id(x) for x in labels["sample"]],
        list(labels["Expression_Subtype"]),
    )

    label_dict = {}
    for i in range(len(sample_ids)):
        if y[i] is None:
            # label_dict[sample_ids[i]] = None
            continue
        label_dict[sample_ids[i]] = y[i]
    y_label_idx = list(set(label_dict.values()))

    _s = set(rna_data.columns)
    _s.intersection_update(list(label_dict.keys()))
    interest_cols = ["attrib_name"]
    interest_cols.extend(list(_s))

    rna_data = rna_data[interest_cols]
    mirna_data = mirna_data[interest_cols]
    meth_data = meth_data[interest_cols]

    for i in label_dict:
        label_dict[i] = y_label_idx.index(label_dict[i])

    y_labels = []
    for _i in interest_cols[1:]:
        y_labels.append(label_dict[_i])
    y_labels = torch.LongTensor(y_labels).to(device)
    return meth_data, mirna_data, rna_data, y_labels


@app.cell
def _(torch):
    def z_score_norm(inp, axis=0):
        mean = torch.mean(inp, axis=axis)
        std = torch.std(inp, axis=axis)

        inp = (inp - mean) / std
        return inp
    return (z_score_norm,)


@app.cell
def _(meth_data, mirna_data, pl, rna_data):
    methy_ids = meth_data["attrib_name"].to_list()
    rna_ids = rna_data["attrib_name"].to_list()
    mirna_ids = mirna_data["attrib_name"].to_list()

    ## Methylation - RNA links
    meth_rna_links = []
    for _i in range(len(methy_ids)):
        if methy_ids[_i] in rna_ids:
            meth_rna_links.append([_i, rna_ids.index(methy_ids[_i])])

    ## RNA - miRNA links
    miRNA_link_df = pl.read_csv(
        "TCGA_LUAD/Data/Homo_sapiens_TarBase-v9.tsv",
        separator="\t",
        null_values=["NA"],
    )
    return (meth_rna_links,)


@app.cell
def _():
    # ct = 0
    # rna_miRNA_links = []
    # for _i in tqdm(range(len(rna_ids))):
    #     _linked_mirna = miRNA_link_df.filter(pl.col("gene_name") == rna_ids[_i])[
    #         "mirna_name"
    #     ]
    #     for _j in _linked_mirna:
    #         if _j.lower() in mirna_ids:
    #             rna_miRNA_links.append([_i, mirna_ids.index(_j.lower())])

    # with open("TCGA_LUAD/vars/rm_links.pkl", 'wb') as file:
    #     pickle.dump(rna_miRNA_links, file)
    return


@app.cell
def _(pickle):
    with open("TCGA_LUAD/vars/rm_links.pkl", "rb") as file:
        rna_miRNA_links = pickle.load(file)
    return (rna_miRNA_links,)


@app.cell
def _(meth_rna_links, np, rna_miRNA_links, torch):
    mr_links = torch.tensor(np.array(meth_rna_links).astype(int)).t().contiguous()
    rm_links = torch.tensor(np.array(rna_miRNA_links).astype(int)).t().contiguous()
    return mr_links, rm_links


@app.cell
def _(HeteroConv, HeteroDictLinear, Linear, LinearNN, SAGEConv, torch):
    class MultiLayerHuman(torch.nn.Module):
        def __init__(self, inp_dim, num_mirna, use_metadata=False, n_metadata=None):
            super().__init__()
            self.lin1 = HeteroDictLinear(
                in_channels=1,
                out_channels=64,
                types=["rna", "methy", "mirna"],
            )
            self.lin2 = HeteroDictLinear(
                in_channels=128, out_channels=128, types=["rna", "methy", "mirna"]
            )

            self.lin3 = HeteroDictLinear(
                in_channels=128,
                out_channels=1,
                types=["rna", "methy", "mirna"],
            )

            self.lin4 = LinearNN(
                in_features=num_mirna,
                out_features=3,
            )

            self.use_metadata = use_metadata
            if self.use_metadata:
                self.metadata_lin = Linear(
                    in_channels=n_metadata,
                    out_channels=128,
                )

            self.norm1 = torch.nn.LayerNorm(64)

            self.conv1 = HeteroConv(
                {
                    ("methy", "controls", "rna"): SAGEConv(64, 128),
                    ("rna", "influences", "mirna"): SAGEConv(64, 128),
                },
                aggr="sum",
            )

            self.norm2 = torch.nn.LayerNorm(128)

            self.drop1 = torch.nn.Dropout(0.4)
            self.drop2 = torch.nn.Dropout(0.4)
            self.drop3 = torch.nn.Dropout(0.4)

            self.conv2 = HeteroConv(
                {
                    ("methy", "controls", "rna"): SAGEConv(16, 64),
                    ("rna", "influences", "mirna"): SAGEConv(16, 64),
                },
                aggr="sum",
            )
            self.norm3 = torch.nn.LayerNorm(64)

            self.conv3 = HeteroConv(
                {
                    ("methy", "controls", "rna"): SAGEConv(64, 64),
                    ("rna", "influences", "mirna"): SAGEConv(64, 64),
                },
                aggr="sum",
            )
            self.norm4 = torch.nn.LayerNorm(64)

        def forward(self, data, edge_dict, metadata=None):
            x_dict = self.lin1(data)
            x_dict = {
                k: self.drop1(self.norm1(v.relu())) for k, v in x_dict.items()
            }

            x_dict = self.conv1(x_dict, edge_dict)
            res1 = self.lin2(x_dict)
            x_dict = {
                k: self.drop2(self.norm2(v).relu()) for k, v in x_dict.items()
            }

            if self.use_metadata:
                x_dict = {
                    k: v + res1[k] + self.metadata_lin(metadata.reshape(1, -1))
                    for k, v in x_dict.items()
                }
            else:
                x_dict = {k: v + res1[k] for k, v in x_dict.items()}

            x_dict = {k: self.drop3(v) for k, v in x_dict.items()}
            x_dict = self.lin3(x_dict)

            mirna_features = x_dict["mirna"].T
            logits = self.lin4(mirna_features)
            # x_dict = {k: F.softplus(v) for k, v in x_dict.items()}

            return x_dict, logits
    return (MultiLayerHuman,)


@app.cell
def _(
    EdgeIndex,
    HeteroData,
    device,
    meth_data,
    mirna_data,
    mr_links,
    np,
    rm_links,
    rna_data,
    torch,
    y_labels,
    z_score_norm,
):
    pyg = HeteroData()
    pyg["rna"].x = z_score_norm(
        torch.FloatTensor(rna_data.to_numpy()[:, 1:].astype(float)), axis=0
    )
    # pyg["rna"].y = z_score_norm(
    #     torch.FloatTensor(rna_data.to_numpy()[:, 1:].astype(float)), axis=0
    # )

    pyg["methy"].x = z_score_norm(
        torch.FloatTensor(meth_data.to_numpy()[:, 1:].astype(float)), axis=0
    )
    # pyg["methy"].y = z_score_norm(
    #     torch.FloatTensor(meth_data.to_numpy()[:, 1:].astype(float)), axis=0
    # )

    pyg["mirna"].x = z_score_norm(
        torch.FloatTensor(mirna_data.to_numpy()[:, 1:].astype(float)), axis=0
    )
    # pyg["mirna"].y = z_score_norm(
    #     torch.FloatTensor(mirna_data.to_numpy()[:, 1:].astype(float)), axis=0
    # )

    pyg["mirna"].y = y_labels

    pyg["methy", "controls", "rna"].edge_index = EdgeIndex(mr_links)
    pyg["rna", "influences", "mirna"].edge_index = EdgeIndex(rm_links)

    train_size = int(0.7 * rna_data.shape[1] - 1)
    train_mask = np.zeros(rna_data.shape[1] - 1)
    train_mask[:train_size] = 1
    np.random.shuffle(train_mask)
    test_mask = np.abs(1 - train_mask)
    train_mask = train_mask.astype(bool)
    test_mask = test_mask.astype(bool)
    pyg["rna"].train_mask = train_mask
    pyg["rna"].test_mask = test_mask

    pyg["methy"].test_mask = train_mask
    pyg["methy"].test_mask = test_mask

    pyg["mirna"].test_mask = train_mask
    pyg["mirna"].test_mask = test_mask

    pyg = pyg.to(device)
    return pyg, test_mask


@app.cell
def _(DataLoader, OmicGraphDataset, device, pyg):
    train_dataset = OmicGraphDataset(pyg, device=device)
    test_dataset = OmicGraphDataset(
        pyg,
        training=False,
        device=device,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)
    return (train_loader,)


@app.cell
def _(MultiLayerHuman, device, pyg, torch):
    model = MultiLayerHuman(
        inp_dim=pyg["rna"].x.shape[1],
        num_mirna=pyg["mirna"].x.shape[0],
        use_metadata=False,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.1, patience=20, min_lr=1e-10
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn, model, optim, scheduler


@app.cell
def _(device, loss_fn, model, optim, scheduler, tqdm, train_loader):
    lrs = []
    losses = []
    losses_val = []
    prog_bar = tqdm(range(10))
    best_model_weights = None
    for epoch in prog_bar:
        for sample_x in train_loader:
            sample_x.to(device)
            model.train()
            optim.zero_grad()
            out = model(sample_x.x_dict, sample_x.edge_index_dict)

            loss1 = loss_fn(out[1], sample_x["mirna"].y[0])

            loss1.backward()
            optim.step()

            model.eval()
            # with torch.no_grad():
            #     out_val = model(sample_x.x_dict, sample_x.edge_index_dict)
            #     # out_val = normalize_output(out_val)

            #     loss1_val = loss_fn(out_val[1][test_mask], y_labels[test_mask])
            # prog_bar.set_description(f"Val Loss:{loss1_val}")
            scheduler.step(loss1)
            # losses_val.append(loss1_val.item())
            losses.append(loss1.detach().item())
            lrs.append(scheduler.get_last_lr())

            # if loss1_val == min(losses_val):
            #     best_model_weights = model.state_dict()
    return best_model_weights, losses, losses_val, lrs


@app.cell
def _():
    return


@app.cell
def _(best_model_weights, model):
    model.load_state_dict(best_model_weights)
    return


@app.cell
def _(losses_val, plt):
    plt.plot(losses_val)
    return


@app.cell
def _(losses, plt):
    plt.plot(losses)
    return


@app.cell
def _(lrs, plt):
    plt.plot(lrs)
    return


@app.cell
def _(lrs):
    lrs[-1]
    return


@app.cell
def _(accuracy_score, out_val, test_mask, torch, y_labels):
    _ypred = torch.argmax(out_val[1], axis=1)[test_mask].cpu()
    _ytrue = y_labels[test_mask].cpu()
    accuracy_score(_ypred, _ytrue)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
