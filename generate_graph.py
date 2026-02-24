import numpy as np
from tqdm import tqdm
import polars as pl
import networkx as nx
import os
from torch_geometric.data import HeteroData
from torch_geometric import EdgeIndex
import torch
from utils import _read_file, _save_file, z_score_norm


def process_ppi_data(inp):
    try:
        prot_edges = [(x["preferredName_A"], x["preferredName_B"]) for x in inp]
        return prot_edges
    except Exception:
        return []


def _get_ppi_edges(org_name="mouse", significant_ppi=False):
    assert org_name == "mouse" or org_name == "human", (
        "Organism name must be 'human' or 'mouse'"
    )

    abs_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "Data", "ppi_files"
    )

    ppi_info_file = os.path.join(abs_path, org_name + "_ppi_info.txt")
    ppi_network_file = os.path.join(abs_path, org_name + "_ppi_network.txt")
    prot_info = pl.read_csv(ppi_info_file, separator="\t")
    ppi_net = pl.read_csv(ppi_network_file, separator=" ")

    significant_ppi_arr = ppi_net.filter(pl.col("combined_score") >= 990)
    prot_map = {}
    for i in prot_info.to_numpy():
        prot_map[i[0]] = i[1]

    renamed_ppi_net = []
    if significant_ppi:
        for i in significant_ppi_arr.to_numpy():
            renamed_ppi_net.append([prot_map[i[0]].upper(), prot_map[i[1]].upper()])
    else:
        for i in ppi_net.to_numpy():
            renamed_ppi_net.append([prot_map[i[0]].upper(), prot_map[i[1]].upper()])
    renamed_ppi_net = pl.DataFrame(renamed_ppi_net, schema=["p1", "p2"])
    return renamed_ppi_net.to_numpy()


def check_repeat_element_edges(arr):
    dup_edges = []
    for i in range(len(arr)):
        if arr[i][0] == arr[i][1]:
            dup_edges.append(i)
    return dup_edges


def generate_random_edges(vals, req_shape, seed=None):
    np.random.seed(seed)
    random_edges = np.random.choice(list(range(len(vals) - 1)), size=req_shape)
    dup_edges = check_repeat_element_edges(random_edges)
    while len(dup_edges) > 0:
        random_edges = np.delete(random_edges, dup_edges, axis=0)
        random_edges = np.append(
            random_edges,
            np.random.choice(list(range(len(vals))), size=(len(dup_edges), 2)),
            axis=0,
        )
        dup_edges = check_repeat_element_edges(random_edges)
    return random_edges


def _generate_graph(
    rna_df: pl.DataFrame,
    prot_df: pl.DataFrame,
    metab_df: pl.DataFrame,
    predefined_network,
    graph_name,
    random_edges,
    n_random_edges_rna,
    n_random_edges_prot,
    use_causal_edges,
    output_dir,
    rna_causal_method,
    prot_causal_method,
    seed=None,
):
    abs_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), output_dir, "network"
    )
    rna_vals_temp = list(rna_df.columns)[1:]
    prot_vals_temp = list(prot_df.columns)[1:]
    metab_vals_temp = list(metab_df.columns)[1:]
    prot_vals_dict = {}
    for i in range(len(prot_vals_temp)):
        prot_vals_temp[i] = prot_vals_temp[i].split(";")
        for prot in prot_vals_temp[i]:
            prot_vals_dict[prot] = i

    rna_vals_dict = {rna_vals_temp[i]: i for i in range(len(rna_vals_temp))}
    metab_vals_dict = {metab_vals_temp[i]: i for i in range(len(metab_vals_temp))}

    rr_edges = []
    rp_edges = []
    pp_edges = []
    pm_edges = []

    random_rna_edges = []
    random_prot_edges = []

    causal_pp_edges = []
    causal_rr_edges = []

    if random_edges:
        print("\tGenerating and adding in random edges")
        random_rna_edges = generate_random_edges(
            rna_df.columns, req_shape=(n_random_edges_rna, 2), seed=seed
        )
        random_prot_edges = generate_random_edges(
            prot_df.columns, req_shape=(n_random_edges_prot, 2), seed=seed
        )

    if use_causal_edges:
        if os.path.exists(
            os.path.join(abs_path, prot_causal_method + "_pp_causal_edges.pkl")
        ):
            print("Causal protein-protein edges found, adding in edges...")
            causal_pp_edges = _read_file(
                os.path.join(abs_path, prot_causal_method + "_pp_causal_edges.pkl")
            )
            causal_pp_edges = [(x[1], x[0]) for x in causal_pp_edges]
        else:
            print("Causal edges files not found... Continuing with base network")

        if os.path.exists(
            os.path.join(abs_path, rna_causal_method + "_rr_causal_edges.pkl")
        ):
            print("Causal RNA-RNA edges found, adding in edges...")
            causal_rr_edges = _read_file(
                os.path.join(abs_path, rna_causal_method + "_rr_causal_edges.pkl")
            )
            causal_rr_edges = [(x[1], x[0]) for x in causal_rr_edges]
        else:
            print("Causal edges files not found... Continuing with base network")

    if (
        os.path.exists(os.path.join(abs_path, "rr_edges.pkl"))
        and os.path.exists(os.path.join(abs_path, "rp_edges.pkl"))
        and os.path.exists(os.path.join(abs_path, "pp_edges.pkl"))
        and os.path.exists(os.path.join(abs_path, "pm_edges.pkl"))
        and predefined_network
    ):
        print("\tEdge files already exist, reading files...")
        rr_e = _read_file(os.path.join(abs_path, "rr_edges.pkl"))
        rp_e = _read_file(os.path.join(abs_path, "rp_edges.pkl"))
        pp_e = _read_file(os.path.join(abs_path, "pp_edges.pkl"))
        pm_e = _read_file(os.path.join(abs_path, "pm_edges.pkl"))

        if len(rr_edges) > 0:
            rr_e = np.append(rr_edges, rr_e, axis=0)
        if len(pp_edges) > 0:
            pp_e = np.append(pp_edges, pp_e, axis=0)
        rr_e = np.unique(rr_e, axis=0)
        pp_e = np.unique(pp_e, axis=0)
        rp_e = np.unique(rp_e, axis=0)
        pm_e = np.unique(pm_e, axis=0)
        return rr_e, rp_e, pp_e, pm_e

    print(
        "\tEdge files not found/Not using predefined network, creating edge files (might take a long time)..."
    )
    g = nx.read_graphml(os.path.join(abs_path, graph_name))
    for i in tqdm(g.edges):
        if i[0] in rna_vals_dict:
            if i[1] in rna_vals_dict:  # RNA-RNA
                rr_edges.append((rna_vals_dict[i[0]], rna_vals_dict[i[1]]))
                rr_edges.append((rna_vals_dict[i[1]], rna_vals_dict[i[0]]))
            elif i[1] in prot_vals_dict:  # RNA-Protein
                if i[1] in prot_vals_dict:
                    rp_edges.append((rna_vals_dict[i[0]], prot_vals_dict[i[1]]))
        else:
            if i[0] not in prot_vals_dict:
                if i[0] in metab_vals_dict:  # Protein-Metabolite
                    if i[1] not in prot_vals_dict:  # Metabolite-Pathway edge, skips...
                        continue
                    pm_edges.append((prot_vals_dict[i[1]], metab_vals_dict[i[0]]))
            else:
                if i[1] in prot_vals_dict:  # Protein-Protein
                    pp_edges.append((prot_vals_dict[i[0]], prot_vals_dict[i[1]]))
                    pp_edges.append((prot_vals_dict[i[1]], prot_vals_dict[i[0]]))
                elif i[1] in metab_vals_dict:  # Protein-Metabolite
                    pm_edges.append((prot_vals_dict[i[0]], metab_vals_dict[i[1]]))
                elif i[1] in rna_vals_dict:  # RNA-Protein
                    if i[0] in prot_vals_dict:
                        rp_edges.append((rna_vals_dict[i[1]], prot_vals_dict[i[0]]))

    rr_edges = np.array(rr_edges).astype(int).reshape((-1, 2))
    rp_edges = np.array(rp_edges).astype(int).reshape((-1, 2))
    pp_edges = np.array(pp_edges).astype(int).reshape((-1, 2))
    pm_edges = np.array(pm_edges).astype(int).reshape((-1, 2))
    if random_edges:
        rr_edges = np.append(rr_edges, np.array(random_rna_edges), axis=0)
        pp_edges = np.append(pp_edges, np.array(random_prot_edges), axis=0)
    if use_causal_edges:
        rr_edges = np.append(rr_edges, np.array(causal_rr_edges), axis=0)
        pp_edges = np.append(pp_edges, np.array(causal_pp_edges), axis=0)
    rr_edges = np.unique(rr_edges, axis=0)
    pp_edges = np.unique(pp_edges, axis=0)
    rp_edges = np.unique(rp_edges, axis=0)
    pm_edges = np.unique(pm_edges, axis=0)
    _save_file(os.path.join(abs_path, "rr_edges.pkl"), rr_edges)
    _save_file(os.path.join(abs_path, "rp_edges.pkl"), rp_edges)
    _save_file(os.path.join(abs_path, "pp_edges.pkl"), pp_edges)
    _save_file(os.path.join(abs_path, "pm_edges.pkl"), pm_edges)
    return rr_edges, rp_edges, pp_edges, pm_edges


def _generate_pyg(
    rna_data,
    prot_data,
    metab_data,
    predefined_network,
    output_dir,
    graph_name,
    random_edges,
    n_random_edges_rna,
    n_random_edges_prot,
    train_test_ratio=0.7,
    use_causal_edges=False,
    rna_causal_method="ges",
    prot_causal_method="fci",
    seed=None,
    device="cpu",
    debug=False,
):
    assert (
        rna_data.shape[0] == prot_data.shape[0]
        and prot_data.shape[0] == metab_data.shape[0]
    ), "Irregular sample sizes for omics"
    assert train_test_ratio > 0 and train_test_ratio <= 1, (
        "Train-test ratio out of bounds (0<r<=1)"
    )

    if type(seed) is list:
        seed = seed[0]

    np.random.seed(seed)

    rr_edges, rp_edges, pp_edges, pm_edges = _generate_graph(
        rna_data,
        prot_data,
        metab_data,
        predefined_network,
        graph_name,
        random_edges,
        n_random_edges_rna,
        n_random_edges_prot,
        use_causal_edges=use_causal_edges,
        output_dir=output_dir,
        rna_causal_method=rna_causal_method,
        prot_causal_method=prot_causal_method,
        seed=seed,
    )

    train_size = int(train_test_ratio * rna_data.shape[0])
    train_mask = np.zeros(rna_data.shape[0])
    train_mask[:train_size] = 1
    np.random.shuffle(train_mask)
    test_mask = np.abs(1 - train_mask)
    train_mask = train_mask.astype(bool)
    test_mask = test_mask.astype(bool)

    prot_inp = np.array(prot_data).astype(np.float64)
    prot_inp = np.nan_to_num(prot_inp, 0.0)
    prot_inp = prot_inp[:, 1:].astype(np.float64)

    metab_inp = np.array(metab_data).astype(np.float64)
    metab_inp = np.nan_to_num(metab_inp, 0.0)
    metab_inp = metab_inp[:, 1:].astype(np.float64)

    rna_inp = np.array(rna_data).astype(np.float64)
    rna_inp = np.nan_to_num(rna_inp, 0.0)
    rna_inp = rna_inp[:, 1:].astype(np.float64)

    pyg = HeteroData()

    # pyg["rna"].x = torch.nn.functional.normalize(torch.FloatTensor(rna_inp.T), dim=1)
    pyg["rna"].x = z_score_norm(torch.FloatTensor(rna_inp.T), axis=0)
    pyg["rna"].train_mask = train_mask
    pyg["rna"].test_mask = test_mask

    pyg["protein"].x = torch.randn(prot_inp.shape[1], prot_inp.shape[0])
    pyg["protein"].y = z_score_norm(torch.FloatTensor(prot_inp.T), axis=0)

    # pyg["protein"].y = torch.nn.functional.normalize(
    #     torch.FloatTensor(prot_inp.T), dim=1
    # )

    pyg["protein"].train_mask = train_mask
    pyg["protein"].test_mask = test_mask

    pyg["metabolite"].x = torch.randn(metab_inp.shape[1], metab_inp.shape[0])
    pyg["metabolite"].y = z_score_norm(torch.FloatTensor(metab_inp.T), axis=0)
    # pyg["metabolite"].y = torch.nn.functional.normalize(
    #     torch.FloatTensor(metab_inp.T), dim=1
    # )

    if torch.isnan(pyg["protein"].y).any():
        pyg["protein"].y = torch.nan_to_num(pyg["protein"].y, nan=0.0)
    if torch.isnan(pyg["metabolite"].y).any():
        pyg["metabolite"].y = torch.nan_to_num(pyg["metabolite"].y, nan=0.0)

    pyg["metabolite"].train_mask = train_mask
    pyg["metabolite"].test_mask = test_mask

    pp_edges = np.array(pp_edges).astype(int)

    rr_edges_t = torch.tensor(rr_edges).t().contiguous()
    pp_edges_t = torch.tensor(pp_edges).t().contiguous()
    pr_edges_t = torch.tensor(rp_edges).t().contiguous()
    pm_edges_t = torch.tensor(pm_edges).t().contiguous()

    pyg["rna", "links", "rna"].edge_index = EdgeIndex(rr_edges_t, is_undirected=False)
    pyg["protein", "interacts", "protein"].edge_index = EdgeIndex(
        pp_edges_t, is_undirected=False
    )

    pyg["rna", "synth", "protein"].edge_index = EdgeIndex(
        pr_edges_t, is_undirected=False
    )
    pyg["protein", "prod", "metabolite"].edge_index = EdgeIndex(
        pm_edges_t, is_undirected=False
    )

    debug and print(
        f"\t RNA-RNA edges = {pyg['rna', 'links', 'rna'].num_edges}. Directed - {pyg['rna', 'links', 'rna'].num_edges == rr_edges_t.shape[1]}\t\t",
    )
    debug and print(
        f"\t Protein-Protein edges = {pyg['protein', 'interacts', 'protein'].num_edges}. Directed - {pyg['protein', 'interacts', 'protein'].num_edges == pp_edges_t.shape[1]}\t\t",
    )
    debug and print(
        f"\t RNA-Protein edges = {pyg['rna', 'synth', 'protein'].num_edges}. Directed - {pyg['rna', 'synth', 'protein'].num_edges == pr_edges_t.shape[1]}\t\t",
    )
    debug and print(
        f"\t Protein-Metabolite edges = {pyg['protein', 'prod', 'metabolite'].num_edges}. Directed - {pyg['protein', 'prod', 'metabolite'].num_edges == pm_edges_t.shape[1]}\t\t",
    )

    pyg.requires_grad_("rna", requires_grad=True)
    pyg.requires_grad_("metabolite", requires_grad=True)
    pyg.requires_grad_("protein", requires_grad=True)
    pyg = pyg.to(device)
    print(pyg)
    return pyg


def _generate_multiple_graphs(
    rna_data,
    prot_data,
    metab_data,
    predefined_network,
    output_dir,
    graph_name,
    random_edges,
    n_random_edges_rna,
    n_random_edges_prot,
    train_test_ratio=0.7,
    use_causal_edges=False,
    rna_causal_method="ges",
    prot_causal_method="fci",
    seed=None,
    device=torch.device("cpu"),
    debug=False,
):
    pyg_graphs = []
    for i in seed:
        assert (
            rna_data.shape[0] == prot_data.shape[0]
            and prot_data.shape[0] == metab_data.shape[0]
        ), "Irregular sample sizes for omics"
        assert train_test_ratio > 0 and train_test_ratio <= 1, (
            "Train-test ratio out of bounds (0<r<=1)"
        )
        np.random.seed(i)

        rr_edges, rp_edges, pp_edges, pm_edges = _generate_graph(
            rna_data,
            prot_data,
            metab_data,
            predefined_network,
            graph_name,
            random_edges,
            n_random_edges_rna,
            n_random_edges_prot,
            use_causal_edges=use_causal_edges,
            output_dir=output_dir,
            rna_causal_method=rna_causal_method,
            prot_causal_method=prot_causal_method,
            seed=i,
        )

        train_size = int(train_test_ratio * rna_data.shape[0])
        train_mask = np.zeros(rna_data.shape[0])
        train_mask[:train_size] = 1
        np.random.shuffle(train_mask)
        test_mask = np.abs(1 - train_mask)
        train_mask = train_mask.astype(bool)
        test_mask = test_mask.astype(bool)

        prot_inp = np.array(prot_data).astype(np.float64)
        prot_inp = np.nan_to_num(prot_inp, 0.0)
        prot_inp = prot_inp[:, 1:].astype(np.float64)

        metab_inp = np.array(metab_data).astype(np.float64)
        metab_inp = np.nan_to_num(metab_inp, 0.0)
        metab_inp = metab_inp[:, 1:].astype(np.float64)

        rna_inp = np.array(rna_data).astype(np.float64)
        rna_inp = np.nan_to_num(rna_inp, 0.0)
        rna_inp = rna_inp[:, 1:].astype(np.float64)

        pyg = HeteroData()

        # pyg["rna"].x = torch.nn.functional.normalize(torch.FloatTensor(rna_inp.T), dim=1)
        pyg["rna"].x = z_score_norm(torch.FloatTensor(rna_inp.T), axis=0)
        pyg["rna"].train_mask = train_mask
        pyg["rna"].test_mask = test_mask

        pyg["protein"].x = torch.randn(prot_inp.shape[1], prot_inp.shape[0])
        pyg["protein"].y = z_score_norm(torch.FloatTensor(prot_inp.T), axis=0)

        # pyg["protein"].y = torch.nn.functional.normalize(
        #     torch.FloatTensor(prot_inp.T), dim=1
        # )

        pyg["protein"].train_mask = train_mask
        pyg["protein"].test_mask = test_mask

        pyg["metabolite"].x = torch.randn(metab_inp.shape[1], metab_inp.shape[0])
        pyg["metabolite"].y = z_score_norm(torch.FloatTensor(metab_inp.T), axis=0)
        # pyg["metabolite"].y = torch.nn.functional.normalize(
        #     torch.FloatTensor(metab_inp.T), dim=1
        # )

        if torch.isnan(pyg["protein"].y).any():
            pyg["protein"].y = torch.nan_to_num(pyg["protein"].y, nan=0.0)
        if torch.isnan(pyg["metabolite"].y).any():
            pyg["metabolite"].y = torch.nan_to_num(pyg["metabolite"].y, nan=0.0)

        pyg["metabolite"].train_mask = train_mask
        pyg["metabolite"].test_mask = test_mask

        pp_edges = np.array(pp_edges).astype(int)

        rr_edges_t = torch.tensor(rr_edges).t().contiguous()
        pp_edges_t = torch.tensor(pp_edges).t().contiguous()
        pr_edges_t = torch.tensor(rp_edges).t().contiguous()
        pm_edges_t = torch.tensor(pm_edges).t().contiguous()

        pyg["rna", "links", "rna"].edge_index = EdgeIndex(
            rr_edges_t, is_undirected=False
        )
        pyg["protein", "interacts", "protein"].edge_index = EdgeIndex(
            pp_edges_t, is_undirected=False
        )

        pyg["rna", "synth", "protein"].edge_index = EdgeIndex(
            pr_edges_t, is_undirected=False
        )
        pyg["protein", "prod", "metabolite"].edge_index = EdgeIndex(
            pm_edges_t, is_undirected=False
        )

        pyg.requires_grad_("rna", requires_grad=True)
        pyg.requires_grad_("metabolite", requires_grad=True)
        pyg.requires_grad_("protein", requires_grad=True)

        pyg = pyg.to(device)
        pyg_graphs.append(pyg)
    debug and print(
        f"\t RNA-RNA edges = {pyg_graphs[0]['rna', 'links', 'rna'].num_edges}. Directed - {pyg_graphs[0]['rna', 'links', 'rna'].num_edges == rr_edges_t.shape[1]}\t\t",
    )
    debug and print(
        f"\t Protein-Protein edges = {pyg_graphs[0]['protein', 'interacts', 'protein'].num_edges}. Directed - {pyg_graphs[0]['protein', 'interacts', 'protein'].num_edges == pp_edges_t.shape[1]}\t\t",
    )
    debug and print(
        f"\t RNA-Protein edges = {pyg_graphs[0]['rna', 'synth', 'protein'].num_edges}. Directed - {pyg_graphs[0]['rna', 'synth', 'protein'].num_edges == pr_edges_t.shape[1]}\t\t",
    )
    debug and print(
        f"\t Protein-Metabolite edges = {pyg_graphs[0]['protein', 'prod', 'metabolite'].num_edges}. Directed - {pyg_graphs[0]['protein', 'prod', 'metabolite'].num_edges == pm_edges_t.shape[1]}\t\t",
    )
    return pyg_graphs
