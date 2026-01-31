from torch_geometric.nn import HeteroConv, HeteroDictLinear, SAGEConv, Linear
import torch


class MultiLayerHuman(torch.nn.Module):
    def __init__(self, inp_dim, use_metadata=False, n_metadata=None):
        super().__init__()
        self.lin1 = HeteroDictLinear(
            in_channels=inp_dim, out_channels=64, types=["rna", "protein", "metabolite"]
        )
        self.lin2 = HeteroDictLinear(
            in_channels=128, out_channels=128, types=["rna", "protein", "metabolite"]
        )

        self.lin3 = HeteroDictLinear(
            in_channels=128,
            out_channels=inp_dim,
            types=["rna", "protein", "metabolite"],
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
                ("rna", "links", "rna"): SAGEConv(64, 128),
                ("protein", "interacts", "protein"): SAGEConv(64, 128),
                ("rna", "synth", "protein"): SAGEConv(64, 128),
                ("protein", "prod", "metabolite"): SAGEConv(64, 128),
            },
            aggr="sum",
        )

        self.norm2 = torch.nn.LayerNorm(128)

        self.drop1 = torch.nn.Dropout(0.4)
        self.drop2 = torch.nn.Dropout(0.4)
        self.drop3 = torch.nn.Dropout(0.4)

        self.conv2 = HeteroConv(
            {
                ("rna", "links", "rna"): SAGEConv(16, 64),
                ("protein", "interacts", "protein"): SAGEConv(16, 64),
                ("rna", "synth", "protein"): SAGEConv(16, 64),
                ("protein", "prod", "metabolite"): SAGEConv(16, 64),
            },
            aggr="sum",
        )
        self.norm3 = torch.nn.LayerNorm(64)

        self.conv3 = HeteroConv(
            {
                ("rna", "links", "rna"): SAGEConv(64, 64),
                ("protein", "interacts", "protein"): SAGEConv(64, 64),
                ("rna", "synth", "protein"): SAGEConv(64, 64),
                ("protein", "prod", "metabolite"): SAGEConv(64, 64),
            },
            aggr="sum",
        )
        self.norm4 = torch.nn.LayerNorm(64)

        # self.conv4 = HeteroConv(
        #     {
        #         ("rna", "links", "rna"): GraphConv(64, 128),
        #         ("protein", "links", "protein"): GraphConv(64, 128),
        #         ("rna", "synth", "protein"): GraphConv(64, 128),
        #         ("protein", "prod", "metabolite"): GraphConv(64, 128),
        #     },
        #     aggr='mean'
        # )
        # self.norm5 = torch.nn.LayerNorm(128)

    def forward(self, data, edge_dict, metadata=None):
        x_dict = self.lin1(data)
        x_dict = {k: self.drop1(self.norm1(v.relu())) for k, v in x_dict.items()}

        x_dict = self.conv1(x_dict, edge_dict)
        res1 = self.lin2(x_dict)
        x_dict = {k: self.drop2(self.norm2(v).relu()) for k, v in x_dict.items()}

        # x_dict = self.conv2(x_dict, edge_dict)
        # x_dict = {k: v + res1[k] for k, v in x_dict.items()}

        # x_dict = {k: self.drop3(self.norm3(v).relu()) for k, v in x_dict.items()}
        # res2 = x_dict

        # x_dict = self.conv3(x_dict, edge_dict)
        # x_dict = {k: v + res2[k] for k, v in x_dict.items()}

        # x_dict = {k: self.norm4(v).relu() for k, v in x_dict.items()}
        
        if self.use_metadata:
            x_dict = {k: v + res1[k] + self.metadata_lin(metadata.reshape(1, -1)) for k, v in x_dict.items()}
        else:
            x_dict = {k: v + res1[k] for k, v in x_dict.items()}
        
        x_dict = {k: self.drop3(v) for k, v in x_dict.items()}
        x_dict = self.lin3(x_dict)
        # x_dict = {k: F.softplus(v) for k, v in x_dict.items()}

        return x_dict