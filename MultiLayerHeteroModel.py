from torch_geometric.nn import HeteroConv, GraphConv, HeteroDictLinear
import torch.nn.functional as F
import torch


class MultiLayerHetero(torch.nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.lin1 = HeteroDictLinear(
            in_channels=inp_dim, out_channels=8, types=["protein", "metabolite"]
        )
        self.norm1 = torch.nn.LayerNorm(8)

        self.conv1 = HeteroConv(
            {
                ("protein", "interacts", "protein"): GraphConv(8, 16),
                ("protein", "links", "metabolite"): GraphConv(8, 16),
            }
        )
        self.norm2 = torch.nn.LayerNorm(16)

        self.drop1 = torch.nn.Dropout(0.4)
        self.drop2 = torch.nn.Dropout(0.4)
        self.drop3 = torch.nn.Dropout(0.4)

        self.conv2 = HeteroConv(
            {
                ("protein", "interacts", "protein"): GraphConv(16, 64),
                ("protein", "links", "metabolite"): GraphConv(16, 64),
            }
        )
        self.norm3 = torch.nn.LayerNorm(64)

        self.conv3 = HeteroConv(
            {
                ("protein", "interacts", "protein"): GraphConv(64, 64),
                ("protein", "links", "metabolite"): GraphConv(64, 64),
            }
        )
        self.norm4 = torch.nn.LayerNorm(64)

        self.conv4 = HeteroConv(
            {
                ("protein", "interacts", "protein"): GraphConv(64, 128),
                ("protein", "links", "metabolite"): GraphConv(64, 128),
            }
        )
        self.norm5 = torch.nn.LayerNorm(128)

        self.lin2 = HeteroDictLinear(
            in_channels=64, out_channels=inp_dim, types=["protein", "metabolite"]
        )

    def forward(self, data, edge_dict):
        
        x_dict = self.lin1(data)
        x_dict = {k: self.drop1(self.norm1(v.relu())) for k, v in x_dict.items()}

        x_dict = self.conv1(x_dict, edge_dict)
        x_dict = {k: self.drop2(self.norm2(v).relu()) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_dict)
        x_dict = {k: self.drop3(self.norm3(v).relu()) for k, v in x_dict.items()}

        x_dict = self.conv3(x_dict, edge_dict)
        x_dict = {k: self.norm4(v).relu() for k, v in x_dict.items()}

        x_dict = self.lin2(x_dict)
        x_dict = {k: F.softplus(v) for k, v in x_dict.items()}
        return x_dict
