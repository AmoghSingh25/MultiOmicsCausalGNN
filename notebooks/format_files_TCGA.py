import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import os
    import pickle

    return os, pickle, pl


@app.cell
def _():
    base_dir = "TCGA_LUAD/Data/"
    return (base_dir,)


@app.cell
def _(base_dir, os, pl):
    _file_names = [
        "RNASeq",
        "Methylation",
        "miRNA",
    ]
    label_file = pl.read_csv(
        os.path.join(base_dir, "luad_subtypes.tsv"), separator="\t"
    )
    _dfs = []
    frmt_dfs = []
    _ids = []
    _start_ids = [1, 1, 1]
    _id_cols = [0, 0, 0]

    for _i in range(len(_file_names)):
        _dfs.append(
            pl.read_excel(
                os.path.join(base_dir, _file_names[_i] + ".xlsx"),
                infer_schema_length=0,
            )
        )
        _ids.append(_dfs[-1].columns[_start_ids[_i] :])

    common_ids = set(_ids[0])
    common_ids.intersection_update(*_ids)
    _label_ids = [x.replace("-", ".")[:-3] for x in label_file["sample"]]
    common_ids.intersection_update(_label_ids)
    common_ids = list(common_ids)
    common_ids.sort()

    for _i in range(len(_file_names)):
        _sel_cols = [_dfs[_i].columns[_id_cols[_i]]]
        _sel_cols.extend(common_ids)
        _frmt_df_i = _dfs[_i][_sel_cols]
        _frmt_df_i.write_csv(os.path.join(base_dir, "frmt_" + _file_names[_i] + ".csv"))
    return (common_ids,)


@app.cell
def _():
    return


@app.cell
def _(pickle):
    with open("Data/input_data/KidneyTumorData/encoded_metadata.pkl", "rb") as file:
        metadata = pickle.load(file)
    return (metadata,)


@app.cell
def _(common_ids, metadata):
    filt_metadata = []
    for i in common_ids:
        _id = i[: i.index("-", 4)]
        if _id in metadata:
            filt_metadata.append(metadata[_id])
        else:
            filt_metadata.append([0] * 1526)
    return (filt_metadata,)


@app.cell
def _(filt_metadata, pickle):
    with open("Data/input_data/KidneyTumorData/frmt_metadata.pkl", "wb") as _file:
        pickle.dump(file=_file, obj=filt_metadata)
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
