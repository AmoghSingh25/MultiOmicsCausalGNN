import marimo

__generated_with = "0.18.0"
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
def _(os, pickle):
    ## Selected genes from https://providers2.genedx.com/Resources/TIS-Files/TIS-B394.pdf
    _s = "BAP1, EPCAM, FH, FLCN, MET, MITF, MLH1, MSH2, MSH6, PMS2, PTEN, SDHB, SDHC, SDHD, TP53, TSC1, TSC2, VHL"
    _sel_genes = [x.strip() for x in _s.split(",")]

    _base_dir = "Data/input_data/KidneyTumorData/"


    def _save_file(path, var):
        with open(path, "wb") as file:
            pickle.dump(var, file)


    _save_file(
        os.path.join(_base_dir, "renal_cancer_gene_filt") + ".pkl", list(_sel_genes)
    )
    return


@app.cell
def _(os, pickle):
    _base_dir = "Data/input_data/KidneyTumorData/"


    def _read_file(path):
        with open(path, "rb") as file:
            myvar = pickle.load(file)
        return myvar


    _var = _read_file(os.path.join(_base_dir, "renal_cancer_gene_filt.pkl"))
    for _i in _var:
        print(_i)
    return


@app.cell
def _(json):
    with open("Data/input_data/KidneyTumorData/gene_list.json", "r") as f:
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

    len(_var)
    return


@app.cell
def _(filt_genes, pl):
    _file_paths = [
        "Data/input_data/KidneyTumorData/transcriptomics.csv",
        "Data/input_data/KidneyTumorData/metabolomics.csv",
        "Data/input_data/KidneyTumorData/proteomics.csv",
    ]
    file_names = [
        "transcriptomics.csv",
        "metabolomics.csv",
        "proteomics.csv",
    ]
    max_null_count = 1
    dfs = []
    cols = []
    _ids = []
    _start_ids = [1, 2, 5]
    _id_cols = [0, 1, 2]

    for _i in range(len(_file_paths)):
        dfs.append(
            pl.read_csv(
                _file_paths[_i], null_values=["Nan", "nan", "N/A", "", "NA"]
            )
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
            "Data/input_data/KidneyTumorData/prot_mapping.tsv", separator="\t"
        )
        _conv_ids = []
        for _i in refseq_ids:
            _search = _prot_mapping.filter(pl.col("From") == _i)[
                "Entry Name"
            ].to_list()
            if len(_search) == 0:
                # print("No Entries for ", _i)
                continue
            _conv_ids.extend(_search)
        return _conv_ids


    _base_dir = "Data/input_data/KidneyTumorData/"

    for _i in [0, 2]:
        filt_idx = np.where(
            dfs[_i].transpose().null_count().to_numpy() < max_null_count
        )
        filt_cols = cols[_i][0][filt_idx[1]]
        if _i == 0:
            filt_cols = list(set(filt_cols).intersection(filt_genes))
        if _i == 2:
            _comp_prots = []
            for _j in filt_cols:
                _comp_prots.extend(_j.split(";"))
            filt_cols = _comp_prots
            filt_cols = _new_id_prot(filt_cols)
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
