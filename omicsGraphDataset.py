import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.utils import dropout_edge


class OmicGraphDataset(Dataset):
    def __init__(
        self,
        input_graph,
        training=True,
        mask=True,
        root=None,
        transform=None,
        pre_transform=None,
        drop_edges=True,
        p=0.5,
        device="mps",
    ):
        super().__init__(root, transform, pre_transform, None)
        ## Data shape - (No. samples, No. features)
        if mask:
            if training:
                self.mask = input_graph["rna"].train_mask
            else:
                self.mask = input_graph["rna"].test_mask
        else:
            self.mask = torch.ones_like(
                torch.FloatTensor(input_graph["rna"].train_mask), dtype=torch.bool
            )

        self.device = device
        self.rna_x = input_graph["rna"].x[:, self.mask]
        self.protein_x = input_graph["protein"].x[:, self.mask]
        self.metab_x = input_graph["metabolite"].x[:, self.mask]
        self.metadata = input_graph['metadata'][:, self.mask]

        self.rna_x.to(self.device)
        self.protein_x.to(self.device)
        self.metab_x.to(self.device)
        self.base_graph = input_graph
        self.p = p
        self.drop_edges = drop_edges
        self.graphs = []
        for i in range(self.len()):
            graph_i = self.gen_graph(i)
            graph_i.to(self.device)
            self.graphs.append(graph_i)

    def drop_edges_func(self, inp_graph):
        pyg_train = inp_graph.clone()
        pyg_train["rna", "links", "rna"].edge_index = dropout_edge(
            inp_graph["rna", "links", "rna"].edge_index, p=self.p
        )[0]
        pyg_train["protein", "interacts", "protein"].edge_index = dropout_edge(
            inp_graph["protein", "interacts", "protein"].edge_index, p=self.p
        )[0]
        pyg_train["rna", "synth", "protein"].edge_index = dropout_edge(
            inp_graph["rna", "synth", "protein"].edge_index, p=self.p
        )[0]
        pyg_train["protein", "prod", "metabolite"].edge_index = dropout_edge(
            inp_graph["protein", "prod", "metabolite"].edge_index, p=self.p
        )[0]
        return pyg_train

    def len(self):
        return self.rna_x.shape[1]
        # return self.rna_x.shape[]

    def get(self, idx):
        return self.graphs[idx]

    def gen_graph(self, idx):
        pyg = HeteroData()
        # pyg = self.base_graph.clone()
        pyg["rna"].x = self.rna_x[:, idx].reshape(-1, 1)
        pyg["protein"].x = torch.randn(self.protein_x.shape[0], 1)
        pyg["metabolite"].x = torch.randn(self.metab_x.shape[0], 1)

        pyg["metadata"] = self.metadata[:, idx].reshape(-1, 1)

        pyg["rna"].y = self.rna_x[:, idx].reshape(-1, 1)
        pyg["protein"].y = torch.nn.functional.normalize(
            torch.tensor(
                self.protein_x[:, idx].T, dtype=torch.float32, device=self.device
            ).reshape(-1, 1),
            dim=0,
        ).to(self.device)
        pyg["metabolite"].y = torch.nn.functional.normalize(
            torch.tensor(
                self.metab_x[:, idx].T, dtype=torch.float32, device=self.device
            ).reshape(-1, 1),
            dim=0,
        ).to(self.device)
        if torch.isnan(pyg["protein"].y).any():
            pyg["protein"].y = torch.nan_to_num(pyg["protein"].y, nan=0.0)
        if torch.isnan(pyg["metabolite"].y).any():
            pyg["metabolite"].y = torch.nan_to_num(pyg["metabolite"].y, nan=0.0)

        pyg["rna", "links", "rna"].edge_index = self.base_graph[
            "rna", "links", "rna"
        ].edge_index
        pyg["protein", "interacts", "protein"].edge_index = self.base_graph[
            "protein", "interacts", "protein"
        ].edge_index
        pyg["rna", "synth", "protein"].edge_index = self.base_graph[
            "rna", "synth", "protein"
        ].edge_index
        pyg["protein", "prod", "metabolite"].edge_index = self.base_graph[
            "protein", "prod", "metabolite"
        ].edge_index

        if self.drop_edges:
            pyg = self.drop_edges_func(pyg)
        return pyg
