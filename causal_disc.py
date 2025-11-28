from utils import _save_file
import numpy as np
import polars as pl
import os
from causal_utils import run_cdnod, run_fci, run_ges, run_pc
import hydra
from omegaconf import DictConfig, OmegaConf
from llm_causal import run_llm
from utils import _read_txt
from sklearn.preprocessing import StandardScaler


def _run_causal_discovery(input_data, all_vals, method="pc", **kwargs):
    # Input data is a Dataframe with columns containing names of the features - proteins, metabolites, RNAs
    # Rows are samples
    print("\tCD method = ", method)
    possible_methods = ["fci", "ges", "pc", "cdnod", "llm"]
    cols = input_data.columns

    scaler = StandardScaler()
    input_data = pl.DataFrame(scaler.fit_transform(input_data.to_numpy()))
    input_data.columns = cols

    assert method in possible_methods, "Invalid causal discovery method"
    try:
        if method == "fci":
            ret = run_fci(input_data, **kwargs)[0].graph
        elif method == "pc":
            ret = run_pc(input_data, **kwargs)[0].graph
        elif method == "cdnod":
            ret = run_cdnod(input_data, **kwargs)[0].graph
        elif method == "ges":
            ret = run_ges(input_data, **kwargs)[0].graph
        elif method == "llm":
            ret = run_llm(
                undirected_edges=[],
                directed_edges=[],
                input_data=input_data,
                objective="cd",
                use_rag=kwargs["use_rag"],
                rag_name=kwargs["rag_name"],
                llm_model_id=kwargs["llm_model_id"],
                temperature=kwargs["llm_temperature"],
                entity_type=kwargs["entity_type"],
                output_dir=kwargs["output_dir"],
            )
    except AssertionError as e:
        print("Causal cache might be outdated... Delete cache folder and try again...")
        raise e
    undirected_edges = []
    directed_edges = []
    edge_idxs = np.where(ret > 0)
    print(edge_idxs)
    sub_vals = input_data.columns
    for i in range(len(edge_idxs[0])):
        edge_i = (sub_vals[edge_idxs[0][i]], sub_vals[edge_idxs[1][i]])
        if (
            ret[edge_idxs[1][i], edge_idxs[0][i]] == 1
            and ret[edge_idxs[0][i], edge_idxs[1][i]] == 1
            and edge_i not in undirected_edges
            and (edge_i[1], edge_i[0]) not in undirected_edges
        ):
            directed_edges.append(edge_i)
            directed_edges.append((edge_i[1], edge_i[0]))
        elif (
            edge_i[1],
            edge_i[0],
        ) not in directed_edges and edge_i not in directed_edges:
            directed_edges.append(
                # edge_idx[1] -> edge_idx[0]
                edge_i
            )
    edge_idxs = np.where(ret < 0)
    for i in range(len(edge_idxs[0])):
        edge_i = (sub_vals[edge_idxs[0][i]], sub_vals[edge_idxs[1][i]])

        if (
            ret[edge_idxs[1][i], edge_idxs[0][i]] == 1
            and (edge_i[1], edge_i[0]) not in directed_edges
            and edge_i not in directed_edges
        ):
            undirected_edges.append(edge_i)
    print("\tNo. of directed edges = ", len(directed_edges))
    print("\tNo. of undirected edges = ", len(undirected_edges))
    print("\tRunning LLM")
    ret = run_llm(
        undirected_edges=undirected_edges,
        directed_edges=directed_edges,
        input_data=input_data,
        objective=None,
        use_rag=kwargs["use_rag"],
        rag_name=kwargs["rag_name"],
        llm_model_id=kwargs["llm_model_id"],
        temperature=kwargs["llm_temperature"],
        entity_type=kwargs["entity_type"],
        output_dir=kwargs["output_dir"],
    )
    del directed_edges, undirected_edges, edge_idxs
    formatted_edges = []
    for i in ret:
        formatted_edges.append(
            (
                all_vals.index(i[0]),
                all_vals.index(i[1]),
            )
        )
    ## Process ret to convert to all labels index and return
    return formatted_edges


def _read_df(path, replaceNan=True, filt_vals=[]):
    df = pl.read_csv(path, null_values=["Nan", "nan", "N/A", "", "NA"])
    df = df.select(sorted(df.columns[1:])).select(pl.all().cast(pl.Float64))
    df_cols = df.columns
    if replaceNan:
        df = df.fill_nan(0).fill_null(0)
    else:
        df = df.drop_nans().drop_nulls()
    filt_vals = set(filt_vals).intersection(
        set(df.columns)
    )  # Removing values not present in the DataFrame
    filt_vals = list(filt_vals)
    filt_vals.sort()
    if len(filt_vals) > 0:
        df = df.select(filt_vals)
    return df, df_cols


def _create_dirs(cfg):
    base_dir = os.path.dirname(os.path.realpath(__file__))
    dirs = [
        os.path.join(base_dir, cfg.output.save_dir),
        os.path.join(base_dir, cfg.output.save_dir, cfg.get("name", "experiment")),
        os.path.join(
            base_dir, cfg.output.save_dir, cfg.get("name", "experiment"), "network"
        ),
        os.path.join(
            base_dir, cfg.output.save_dir, cfg.get("name", "experiment"), "causal_cache"
        ),
    ]
    for i in dirs:
        if not os.path.exists(i):
            os.mkdir(i)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    print("Configuration - ")
    print(OmegaConf.to_yaml(cfg))

    _create_dirs(cfg)
    data_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        cfg.output.save_dir,
        cfg.get("name", "experiment"),
        "network",
    )
    cache_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        cfg.output.save_dir,
        cfg.get("name", "experiment"),
        "causal_cache",
    )

    print("Running Causal Discovery on RNA data...")
    if cfg.causal.filt_rna is None:
        print("\tNo filtered RNA file given...")
        if cfg.causal.rna_keep is None:
            print("\tNo filtered RNA list given in config...Using all RNAs")
            rna_filt = []
        else:
            rna_filt = cfg.causal.rna_keep
    else:
        rna_filt = _read_txt(cfg.causal.filt_rna)
    rna_filt = list(rna_filt)
    print("\tNumber of RNAs to keep = ", len(rna_filt))

    rna_df, rna_vals = _read_df(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg.data.rna_data),
        replaceNan=cfg.causal.replaceNan,
        filt_vals=rna_filt,  # RNA IDs that must be kept for causal detection, others will be removed
    )
    print("\tProcessed RNA shape - ", rna_df.shape)

    # If this raises AssertionError Cache mismatch, delete the Causal Discovery cache files
    rna_edges = _run_causal_discovery(
        input_data=rna_df,
        method=cfg.causal.rna_method,
        all_vals=rna_vals,
        indep_test=cfg.causal.rna_indep_test,
        cache_path=os.path.join(cache_dir, f"rna_cd_{cfg.causal.rna_method}.json"),
        llm_objective=cfg.causal.llm.get("objective", None),
        llm_model_id=cfg.causal.llm.get("model_id", None),
        llm_temperature=cfg.causal.llm.get("temperature", 0.7),
        use_rag=cfg.causal.llm.get("rag", False),
        rag_name=cfg.get("name", "experiment") + "_rna",
        entity_type="RNA",
        output_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            cfg.output.save_dir,
            cfg.get("name", "experiment"),
        ),
        alpha=cfg.causal.get("alpha", 0.05),
    )
    _save_file(
        os.path.join(data_dir, cfg.causal.rna_method + "_rr_causal_edges.pkl"),
        rna_edges,
    )

    print("Running Causal Discovery on Protein data...")
    if cfg.causal.filt_prot is None:
        print("\tNo filtered Proteins file given...")
        if cfg.causal.prot_keep is None:
            print("\tNo filtered Protein list given in config...Using all Proteins")
            prot_filt = []
        else:
            prot_filt = cfg.causal.prot_keep
    else:
        prot_filt = _read_txt(cfg.causal.filt_prot)
    prot_filt = list(prot_filt)
    print("\tNumber of Proteins to keep = ", len(prot_filt))

    prot_df, prot_vals = _read_df(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg.data.prot_data),
        replaceNan=cfg.causal.replaceNan,
        filt_vals=prot_filt,  # Protein IDs that must be kept for causal detection, others will be removed
    )

    # If this raises AssertionError Cache mismatch, delete the Causal Discovery cache files
    prot_edges = _run_causal_discovery(
        input_data=prot_df,
        method=cfg.causal.prot_method,
        all_vals=prot_vals,
        indep_test=cfg.causal.rna_indep_test,
        cache_path=os.path.join(cache_dir, f"prot_cd_{cfg.causal.prot_method}.json"),
        llm_objective=cfg.causal.llm.get("objective", None),
        llm_model_id=cfg.causal.llm.get("model_id", None),
        llm_temperature=cfg.causal.llm.get("temperature", 0.7),
        use_rag=cfg.causal.llm.get("rag", False),
        rag_name=cfg.get("name", "experiment") + "_prot",
        entity_type="Protein",
        output_dir=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            cfg.output.save_dir,
            cfg.get("name", "experiment"),
        ),
        alpha=cfg.causal.get("alpha", 0.05),
    )
    _save_file(
        os.path.join(data_dir, cfg.causal.prot_method + "_pp_causal_edges.pkl"),
        prot_edges,
    )


if __name__ == "__main__":
    main()
