import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import polars as pl
    return np, pl


@app.cell
def _(pl):
    clustering_nb = pl.read_excel(
        "Data/input_data/KidneyTumorData/DAVID_clustering.xlsx"
    )
    return (clustering_nb,)


@app.cell
def _(clustering_nb):
    clustering_nb
    return


@app.cell
def _(clustering_nb, np):
    def generate_genes(gene_inp):
        gene_set = set()
        for i in gene_inp:
            genes = [x.strip() for x in i.split(",")]
            gene_set.update(genes)
        gene_list = list(gene_set)
        gene_edges = []
        for _i in range(len(gene_list)):
            for _j in range(_i + 1, len(gene_list)):
                gene_edges.append([gene_list[_i], gene_list[_j]])
                gene_edges.append([gene_list[_j], gene_list[_i]])
        gene_edges = np.array(gene_edges)
        return gene_set, gene_edges

    clustering_filt = clustering_nb[:, 5].to_numpy()
    clustering_filt = list(clustering_filt[clustering_filt is not None])
    start = clustering_filt.index("Genes")
    all_genes = set()
    gene_edges = []

    while True:
        if "Genes" in clustering_filt[start + 1 :]:
            end_idx = clustering_filt.index("Genes", start + 1)
        else:
            end_idx = len(clustering_filt)
        genes_i = clustering_filt[start + 1 : end_idx]
        gene_set_i, gene_edges_i = generate_genes(genes_i)
        all_genes.update(gene_set_i)
        gene_edges.extend(gene_edges_i)

        clustering_filt = clustering_filt[end_idx:]
        if "Genes" not in clustering_filt:
            break
        start = clustering_filt.index("Genes")

    gene_edges = np.unique(np.array(gene_edges), axis=0)
    return all_genes, gene_edges


@app.cell
def _(pl):
    rna_cols = pl.read_csv(
        "Data/input_data/KidneyTumorData/frmt_transcriptomics.csv"
    ).columns
    prot_cols = pl.read_csv(
        "Data/input_data/KidneyTumorData/frmt_proteomics.csv"
    ).columns
    prot_vals_dict = {}
    for _i in range(len(prot_cols)):
        prot_vals_temp_i = prot_cols[_i].split(";")
        for prot in prot_vals_temp_i:
            prot_vals_dict[prot] = _i
    return prot_vals_dict, rna_cols


@app.cell
def _(gene_edges, np, rna_cols):
    gene_edges_idx = []
    for _i in range(len(gene_edges)):
        gene_edges_idx.append(
            [rna_cols.index(gene_edges[_i][0]), rna_cols.index(gene_edges[_i][1])]
        )
    gene_edges_idx = np.array(gene_edges_idx)
    return (gene_edges_idx,)


@app.cell
def _(gene_edges_idx):
    from utils import _save_file
    _save_file("output/rcc_bn3/network/rr_edges.pkl", gene_edges_idx)
    return


@app.cell
def _(pl):
    gene_prot_mapping = pl.read_csv(
        "Data/input_data/KidneyTumorData/DAVID_prot_mapping.tsv", separator="\t"
    )
    return (gene_prot_mapping,)


@app.cell
def _(all_genes):
    all_genes
    return


@app.cell
def _(gene_prot_mapping):
    gene_prot_mapping
    return


@app.cell
def _(gene_edges, gene_prot_mapping, np, prot_vals_dict):
    prot_names = gene_prot_mapping["Entry Name"].to_list()
    gene_names = gene_prot_mapping["From"].to_list()
    prot_edges_index = []

    def get_prot_idx(gene_id):
        if gene_id not in gene_names:
            return -1
        _prot_i = prot_names[gene_names.index(gene_id)]
        if _prot_i in prot_vals_dict:
            return prot_vals_dict[_prot_i]
        else:
            return -1

    for _i in range(len(gene_edges)):
        _node1 = get_prot_idx(gene_edges[_i][0])
        _node2 = get_prot_idx(gene_edges[_i][1])

        if _node1 == -1 or _node2 == -1:
            continue
        prot_edges_index.append([_node1, _node2])
    prot_edges_index = np.array(prot_edges_index)
    return (prot_edges_index,)


@app.cell
def _(prot_edges_index):
    prot_edges_index
    return


@app.cell
def _(prot_edges_index):
    from utils import _save_file
    _save_file("output/rcc_bn3/network/pp_edges.pkl", prot_edges_index)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
