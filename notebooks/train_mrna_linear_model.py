import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import polars as pl
    import torch
    from torch.nn import Linear, Conv1d, Dropout
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import lightning as L
    from torchmetrics import Accuracy, AUROC

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import accuracy_score
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch import Trainer
    return (
        AUROC,
        Accuracy,
        DataLoader,
        Dropout,
        F,
        L,
        Linear,
        TensorDataset,
        WandbLogger,
        accuracy_score,
        np,
        pl,
        plt,
        torch,
    )


@app.cell
def _(pl, torch):
    def frmt_id(x):
        return x.replace("-", ".")[:-3]


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


    def z_score_norm(inp, axis=0):
        mean = torch.mean(inp, axis=axis)
        std = torch.std(inp, axis=axis)

        inp = (inp - mean) / std
        return inp


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
    return rna_data, y_labels


@app.cell
def _(Dropout, F, Linear, torch):
    class Model(torch.nn.Module):
        def __init__(self, inp_dim):
            super().__init__()
            self.lin1 = Linear(in_features=inp_dim, out_features=256)
            # self.conv1 = Conv1d(in_channels=256, out_channels=128, kernel_size=1)
            self.lin2 = Linear(in_features=256, out_features=512)
            self.lin3 = Linear(in_features=512, out_features=3)
            self.drop1 = Dropout(p=0.3)
            self.drop2 = Dropout(p=0.3)

        def forward(self, x):
            x = self.lin1(x)
            x = F.relu(x)
            # x = self.drop1(x)
            x = self.lin2(x)
            x = F.relu(x)
            # x = self.drop2(x)
            # x = self.conv1(x)
            # x = F.relu(x)
            x = self.lin3(x)
            return x
    return (Model,)


@app.cell
def _(AUROC, Accuracy, L, torch):
    class LitMLP(L.LightningModule):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp
            self.acc = Accuracy(task="multiclass", num_classes=3)
            self.auc = AUROC(task="multiclass", num_classes=3)

        def training_step(self, batch, batch_idx):
            loss, acc, auc = self._get_preds_loss_accuracy(batch)

            self.log("train_loss", loss, on_epoch=True, on_step=False)
            self.log("train_accuracy", acc, on_epoch=True, on_step=False)
            self.log("train_auc", auc, on_epoch=True, on_step=False)
            sch = self.lr_schedulers()
            sch.step(loss)
            return loss

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            loss, acc, auc = self._get_preds_loss_accuracy(batch)

            self.log("val_loss", loss, on_epoch=True, on_step=False)
            self.log("val_accuracy", acc, on_epoch=True, on_step=False)
            self.log("val_auc", auc, on_epoch=True, on_step=False)
            return loss

        def _get_preds_loss_accuracy(self, batch):
            x, y = batch
            y_pred = self.mlp(x).reshape(-1, 3)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(y_pred, y)

            acc = self.acc(torch.argmax(y_pred, axis=1), y)
            auc = self.auc(y_pred, y)
            return loss, acc, auc

        def configure_optimizers(self):
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-12
            )
            monitor = "train_loss"

            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.scheduler, "monitor": monitor},
            }
    return (LitMLP,)


@app.cell
def _(LitMLP, Model, WandbLogger, np, rna_data, torch, y_labels):
    x = torch.FloatTensor(rna_data.to_numpy()[:, 1:].astype(float)).T

    train_size = int(0.7 * x.shape[0])
    train_mask = np.zeros(x.shape[0])
    train_mask[:train_size] = 1
    np.random.shuffle(train_mask)
    test_mask = np.abs(1 - train_mask)
    train_mask = train_mask.astype(bool)
    test_mask = test_mask.astype(bool)

    x_train = x[train_mask]
    x_test = x[test_mask]

    # model = Model(x.shape[1])
    model = Model(x.shape[1])
    mlp = LitMLP(model)

    wandb_logger = WandbLogger(project="TCGA_LUAD_MLP", name="epoch_10k")

    y_train = torch.LongTensor(y_labels)[train_mask].reshape(-1, 1)
    y_test = torch.LongTensor(y_labels)[test_mask].reshape(-1, 1)
    return (
        mlp,
        model,
        test_mask,
        wandb_logger,
        x_test,
        x_train,
        y_test,
        y_train,
    )


@app.cell
def _(
    DataLoader,
    L,
    TensorDataset,
    mlp,
    wandb_logger,
    x_test,
    x_train,
    y_test,
    y_train,
):
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_test, y_test)

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=13,
    #     persistent_workers=True,
    # )
    val_loader = DataLoader(
        val_ds, batch_size=32, num_workers=13, persistent_workers=True
    )

    trainer = L.Trainer(max_epochs=2000, logger=[wandb_logger])
    trainer.fit(
        model=mlp,
        train_dataloaders=train_ds,
        val_dataloaders=val_ds,
    )
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Archive
    """)
    return


@app.cell
def _():
    # best_weights = None
    # lrs = []
    # train_losses = []
    # val_losses = []
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-12
    # )
    # for t in tqdm(range(10000)):
    #     model.train()
    #     y_pred = model(x_train)
    #     loss = loss_fn(y_pred, y_train)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     model.eval()
    #     if t % 300 == 0:
    #         with torch.no_grad():
    #             y_pred_val = model(x_test)
    #             val_loss = loss_fn(y_pred_val, y_test)
    #             val_losses.append(val_loss)
    #             if val_loss == min(val_losses):
    #                 best_weights = model.state_dict()
    #     optimizer.step()
    #     scheduler.step(loss.item())
    #     train_losses.append(loss.item())
    #     lrs.append(scheduler.get_last_lr()[0])
    return


@app.cell
def _(plt, val_losses):
    plt.plot(val_losses)
    return


@app.cell
def _(plt, train_losses):
    plt.plot(train_losses)
    return


@app.cell
def _(lrs, plt):
    plt.plot(lrs)
    return


@app.cell
def _(accuracy_score, np, test_mask, y_labels, y_pred_val):
    accuracy_score(y_pred_val.argmax(axis=1), np.array(y_labels)[test_mask])
    return


@app.cell
def _(best_weights, model):
    model.load_state_dict(best_weights)
    return


@app.cell
def _(accuracy_score, model, np, test_mask, x_test, y_labels):
    _y_pred_val = model(x_test)
    accuracy_score(_y_pred_val.argmax(axis=1), np.array(y_labels)[test_mask])
    return


@app.cell
def _(best_weights, torch):
    torch.save(best_weights, "TCGA_LUAD/saved_weights/mlp_94acc.pt")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
