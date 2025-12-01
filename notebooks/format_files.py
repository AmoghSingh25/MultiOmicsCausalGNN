import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    _file_paths = [
        "Data/input_data/KidneyTumorData/transcriptomics.csv",
        "Data/input_data/KidneyTumorData/metabolomics.csv",
        "Data/input_data/KidneyTumorData/proteomics.csv",
    ]
    _file_names = [
        "transcriptomics.csv",
        "metabolomics.csv",
        "proteomics.csv",
    ]
    _dfs = []
    frmt_dfs = []
    _ids = []
    _start_ids = [1, 2, 5]
    _id_cols = [0, 1, 2]

    for _i in range(len(_file_paths)):
        _dfs.append(
            pl.read_csv(_file_paths[_i], infer_schema_length=0, null_values=["NA"])
        )
        _ids.append(_dfs[-1].columns[_start_ids[_i] :])

    common_ids = set(_ids[0])
    common_ids.intersection_update(*_ids)
    common_ids = list(common_ids)
    common_ids.sort()

    for _i in range(len(_file_paths)):
        _sel_cols = _dfs[_i].columns[: _start_ids[_i]]
        _sel_cols.extend(common_ids)
        _df_i = _dfs[_i].select(_sel_cols)
        frmt_dfs.append(_df_i)

    _names = []
    for _i in range(len(_file_paths)):
        _names.append(frmt_dfs[_i].columns[_start_ids[_i] :])

    for _i in range(0, len(frmt_dfs)):
        if _i != 3:
            print(_file_names[_i])
            _sel_ids = [frmt_dfs[_i].columns[_id_cols[_i]]]
            _sel_ids.extend(frmt_dfs[_i].columns[_start_ids[_i] :])
            frmt_dfs[_i] = frmt_dfs[_i].drop_nulls(
                subset=frmt_dfs[_i].columns[_id_cols[_i]]
            )
            frmt_dfs[_i] = frmt_dfs[_i].select(_sel_ids).transpose()
            frmt_dfs[_i].columns = frmt_dfs[_i].row(0)
            _cols = frmt_dfs[_i].row(0)
            frmt_dfs[_i] = frmt_dfs[_i].slice(1)
        frmt_dfs[_i].write_csv(
            "Data/input_data/KidneyTumorData/frmt_" + _file_names[_i]
        )
    return (common_ids,)


@app.cell
def _(common_ids, pl):
    _t = pl.read_excel(
        "/Users/amogh/Downloads/mmc2.xlsx", sheet_name="clinical_characteristics"
    )
    _t = _t.select(["caseID", "tumorClass"])
    _t = _t.filter(pl.col("caseID").is_in(common_ids))
    _t
    return


@app.cell
def _(common_ids, pl):
    _prot_mapping = pl.read_csv(
        "Data/input_data/KidneyTumorData/prot_mapping.tsv", separator="\t"
    )
    _map_dict = {
        _i: _j for _i, _j in zip(_prot_mapping["From"], _prot_mapping["Entry Name"])
    }
    _prot_df = pl.read_csv("Data/input_data/KidneyTumorData/proteomics.csv")
    _prot_ids = []
    for _i in _prot_df["ProteinID"]:
        if _map_dict.get(_i):
            _prot_ids.append(_map_dict[_i])
        else:
            _prot_ids.append(None)

    _prot_df = _prot_df.with_columns(pl.Series("ProtID", _prot_ids))
    _ids = ["ProtID"]
    _ids.extend(common_ids)
    _prot_df = _prot_df.select(_ids)
    _prot_df = (
        _prot_df.unique(subset="ProtID")
        .drop_nulls(subset=_prot_df.columns[0])
        .transpose()
    )
    _prot_df.columns = _prot_df.row(0)
    _prot_df = _prot_df.slice(1)
    print(_prot_df.shape)
    _prot_df.write_csv("Data/input_data/KidneyTumorData/frmt_proteomics.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
