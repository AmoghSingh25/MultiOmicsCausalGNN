import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy import stats
    from tqdm import tqdm
    from statsmodels.stats.multitest import multipletests

    return mo, multipletests, np, pl, stats, tqdm


@app.cell
def _(pl):
    _file_paths = [
        "Data/input_data/KidneyTumorData/transcriptomics.csv",
        "Data/input_data/KidneyTumorData/proteomics.csv",
    ]

    start_ids = [1, 5]
    id_cols = ["ID", "ProteinID"]

    dfs = []
    for _i in range(len(_file_paths)):
        dfs.append(
            pl.read_csv(_file_paths[_i], infer_schema_length=0, null_values=["NA"])
        )
    return dfs, id_cols, start_ids


@app.cell
def _(dfs, id_cols, multipletests, np, start_ids, stats, tqdm):
    ## Transcriptomics
    def return_filt(_i):
        pvals = []
        logfc = []

        _ids = dfs[1].columns[start_ids[1] :]
        _ids = [x for x in _ids if x.startswith("C") and "-" in x]
        norm_samples = [x for x in _ids if x.endswith("-N")]
        tumor_samples = [x for x in _ids if x.endswith("-T")]

        norm_df = dfs[1][[id_cols[1], *norm_samples]]
        tumor_df = dfs[1][[id_cols[1], *tumor_samples]]

        for _j in tqdm(range(norm_df.shape[0])):
            _gene_id = norm_df[_j].to_numpy()[0, 0]
            _p1 = norm_df[_j].to_numpy()[0, 1:].astype(np.float64)
            _p2 = tumor_df[_j].to_numpy()[0, 1:].astype(np.float64)
            stat_j = stats.ttest_ind(_p1, _p2, equal_var=True)
            pvals.append(stat_j.pvalue)
            logfc.append(_p2.mean() - _p1.mean())
        # ent_ids = dfs[1][id_cols[1]].to_list()
        _pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
        fdrs = multipletests(_pvals, method="fdr_bh")
        _a = np.where(np.abs(np.array(logfc)) > 1)[0]
        _b = np.where(fdrs[1] < 0.05)[0]

        _s = set(_a)
        _s.intersection_update(_b)
        return list(_s)

    return (return_filt,)


@app.cell
def _(dfs, return_filt):
    _r = return_filt(0)
    gene_list = dfs[0]["ID"][_r].to_list()

    with open(
        "Data/input_data/KidneyTumorData/renal_cancer_gene_filt.txt", "r"
    ) as _file:
        gene_list_rcc = _file.readlines()
    gene_list.extend(gene_list_rcc)
    gene_list = [x.strip() for x in list(set(gene_list))]
    with open(
        "Data/input_data/KidneyTumorData/transcriptomics_filt_edges.txt", "w"
    ) as _file:
        _file.write("\n".join(gene_list))

    # with open(
    #     "Data/input_data/KidneyTumorData/transcriptomics_filt_edges.txt", "r"
    # ) as _file:
    #     gene_list_2 = _file.read()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Proteomics
    """)
    return


@app.cell
def _(dfs, pl, return_filt):
    _r = return_filt(1)
    prot_list = dfs[1]["ProteinID"][_r].to_list()

    with open(
        "Data/input_data/KidneyTumorData/renal_cancer_prot_filt.txt", "r"
    ) as _file:
        prot_list_rcc = _file.readlines()
    prot_list.extend(prot_list_rcc)
    prot_list = [x.strip() for x in list(set(prot_list))]

    def _new_id_prot(refseq_ids):
        _prot_mapping = pl.read_csv(
            "Data/input_data/KidneyTumorData/prot_mapping.tsv", separator="\t"
        )
        _conv_ids = []
        for _i in refseq_ids:
            _search = _prot_mapping.filter(pl.col("From") == _i)["Entry Name"].to_list()
            if len(_search) == 0:
                # print("No Entries for ", _i)
                continue
            _conv_ids.extend(_search)
        return _conv_ids

    conv_ids = _new_id_prot(prot_list)

    with open(
        "Data/input_data/KidneyTumorData/proteomics_filt_edges.txt", "w"
    ) as _file:
        _file.write("\n".join(conv_ids))

    # with open(
    #     "Data/input_data/KidneyTumorData/proteomics_filt_edges.txt", "r"
    # ) as _file:
    #     prot_list_2 = _file.read()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
