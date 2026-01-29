import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import json
    import polars as pl
    from sklearn.preprocessing import OneHotEncoder
    import pickle
    import matplotlib.pyplot as plt
    return OneHotEncoder, np, pickle, pl


@app.cell
def _(pl):
    clinical_metadata_df = pl.read_csv(
        "Data/input_data/KidneyTumorData/orig_data/clinical.tsv",
        separator="\t",
        ignore_errors=True,
    )
    exposure_metadata_df = pl.read_csv(
        "Data/input_data/KidneyTumorData/orig_data/exposure.tsv",
        separator="\t",
        ignore_errors=True,
    )
    return clinical_metadata_df, exposure_metadata_df


@app.cell
def _(clinical_metadata_df, exposure_metadata_df):
    _interest_cols_1 = [
        "cases.submitter_id",
        "demographic.gender",
        "diagnoses.age_at_diagnosis",
    ]
    _interest_cols_2 = [
        "cases.submitter_id",
        "exposures.alcohol_history",
        "exposures.alcohol_intensity",
        "exposures.tobacco_smoking_status",
    ]

    features_1 = clinical_metadata_df[_interest_cols_1]
    features_2 = exposure_metadata_df[_interest_cols_2]
    return features_1, features_2


@app.cell
def _(features_2):
    features_2
    return


@app.cell
def _(features_1, features_2, np):
    metadata_dict = {}
    for _i in range(len(features_1)):
        _id = features_1[_i, 0]
        if metadata_dict.get(_id) is None:
            metadata_dict[_id] = features_1[_i, 1:].to_numpy()[0].tolist()

    for _i in range(len(features_2)):
        _id = features_2[_i, 0]
        if metadata_dict.get(_id) is not None:
            metadata_dict[_id].extend(features_2[_i, 1:].to_numpy()[0].tolist())

    combined_features = []
    for _i in metadata_dict:
        if len(metadata_dict[_i]) == 5:
            combined_features.append(metadata_dict[_i])
        else:
            combined_features.append([None, None, None, None, None])
    combined_features = np.array(combined_features)
    return combined_features, metadata_dict


@app.cell
def _(OneHotEncoder):
    enc = OneHotEncoder(handle_unknown="ignore")
    return (enc,)


@app.cell
def _(combined_features, enc):
    ret = enc.fit_transform(combined_features).toarray()
    return (ret,)


@app.cell
def _(ret):
    ret
    return


@app.cell
def _(metadata_dict, ret):
    mapped_metadata_dict = {}
    for i in range(len(metadata_dict)):
        mapped_metadata_dict[list(metadata_dict.keys())[i]] = ret[i]
    return (mapped_metadata_dict,)


@app.cell
def _(mapped_metadata_dict, pickle):
    with open("Data/input_data/KidneyTumorData/encoded_metadata.pkl", "wb") as file:
        pickle.dump(mapped_metadata_dict, file)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
