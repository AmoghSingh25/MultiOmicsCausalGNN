import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import pickle
    import numpy as np
    import polars as pl
    import json

    return json, np, os, pickle, pl


@app.cell
def _(json):
    with open("Data/input_data/data2/gene_list.json", "r") as f:
        filt = json.load(f)

    filt_genes = set()
    for _i in filt:
        filt_genes.update(filt[_i]["geneSymbols"])
    return (filt_genes,)


@app.cell
def _(pickle):
    def _read_file(path):
        with open(path, "rb") as file:
            var = pickle.load(file)
        return var

    _var = _read_file("Data/input_data/data2/proteomics_filt_edges.pkl")

    print(_var[:10])
    return


@app.cell
def _(filt_genes, pl):
    _file_paths = [
        "Data/input_data/data2/circular_rna.csv",
        "Data/input_data/data2/gene_expression.csv",
        "Data/input_data/data2/metabolomics.csv",
        "Data/input_data/data2/proteomics.csv",
    ]
    file_names = [
        "circular_rna.csv",
        "gene_expression.csv",
        "metabolomics.csv",
        "proteomics.csv",
    ]
    max_null_count = 1
    dfs = []
    cols = []
    _ids = []
    _start_ids = [4, 4, 2, 3]
    _id_cols = [2, 1, 1, 1]

    for _i in range(len(_file_paths)):
        dfs.append(
            pl.read_csv(_file_paths[_i], null_values=["Nan", "nan", "N/A", "", "NA"])
        )
        if _i == 3:
            print(dfs[_i].head())
            dfs[_i] = dfs[_i].filter(pl.col("symbol").is_in(filt_genes))
        cols.append(list(dfs[_i].select(dfs[_i].columns[_id_cols[_i]])))
        dfs[_i] = dfs[_i].select(list(dfs[_i].columns)[_start_ids[_i] :])
    return cols, dfs, file_names, max_null_count


@app.cell
def _(cols, dfs, file_names, filt_genes, max_null_count, np, os, pickle, pl):
    def _save_file(path, var):
        with open(path, "wb") as file:
            pickle.dump(var, file)

    def _save_txt(path, _l):
        _file = open(path, "w")
        for _i in _l:
            _file.write(_i + "\n")
        _file.close()

    def _new_id_prot(refseq_ids):
        _prot_mapping = pl.read_csv(
            "Data/input_data/data2/prot_mapping.tsv", separator="\t"
        )
        _conv_ids = []
        for _i in refseq_ids:
            _search = _prot_mapping.filter(pl.col("From") == _i)["Entry Name"].to_list()
            if len(_search) == 0:
                print("No Entries for ", _i)
                continue
            _conv_ids.extend(_search)
        return _conv_ids

    _base_dir = "Data/input_data/data2"

    for _i in [0, 1, 3]:
        filt_idx = np.where(
            dfs[_i].transpose().null_count().to_numpy() < max_null_count
        )
        filt_cols = cols[_i][0][filt_idx[1]]
        if _i == 1:
            filt_cols = list(set(filt_cols).intersection(filt_genes))
        if _i == 3:
            filt_cols = _new_id_prot(filt_cols)
            # filt_cols = _mapped_cols
        filt_cols = set(filt_cols)
        if None in filt_cols:
            filt_cols.remove(None)
        print(file_names[_i])
        print(len(list(filt_cols)))
        _file_name = file_names[_i][: file_names[_i].index(".")] + "_filt_edges"
        _save_file(os.path.join(_base_dir, _file_name) + ".pkl", list(filt_cols))
        _save_txt(os.path.join(_base_dir, _file_name) + ".txt", list(filt_cols))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
