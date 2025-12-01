# Process
# Read data in DFs for omics
# Run causal discovery if chosen
# Create the graph of omics to connect the omics together - Initially use networkx graphs and then move to manually imbibing links from CD and PPIs - Speed up creation of the graph using sets or pre-storing
# Run training of the CausalGNN on the graph
# Pathway analysis on average and per sample if required
# Perform interventions on the learnt graph

from train_causalGNN import _trainGNN
from train_cGNN_combined import _trainGNN as _trainGNN_combined
from generate_graph import _generate_pyg
import polars as pl
from analysis import _pathway_analysis, _intervention_analysis
import os
import copy
from generate_pathways import _generate_base_network
import hydra
from omegaconf import DictConfig, OmegaConf


def _create_dirs(cfg):
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    exp_name = cfg.get("name", "experiment")
    dirs = [
        os.path.join(parent_dir, cfg.output.save_dir),
        os.path.join(parent_dir, cfg.output.save_dir, "logs"),
        os.path.join(parent_dir, cfg.output.save_dir, exp_name),
        os.path.join(parent_dir, cfg.output.save_dir, exp_name, "network"),
        os.path.join(parent_dir, cfg.output.save_dir, exp_name, "weights"),
        os.path.join(parent_dir, cfg.output.save_dir, exp_name, "figures"),
    ]
    for i in dirs:
        if not os.path.exists(i):
            os.mkdir(i)
    return os.path.join(parent_dir, cfg.output.save_dir, exp_name)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    print("Configuration - ")
    print(OmegaConf.to_yaml(cfg))
    output_dir = _create_dirs(cfg)
    rna_df = pl.read_csv(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg.data.rna_data),
        null_values=["NA"],
    )
    metab_df = pl.read_csv(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg.data.metab_data),
        null_values=["NA"],
    )
    prot_df = pl.read_csv(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg.data.prot_data),
        null_values=["NA"],
    )
    _generate_base_network(
        rna_df,
        predefined_network=cfg.data.predefined_network,
        graph_name=cfg.data.get("base_network_file", "human_base_network.graphml"),
        output_dir=output_dir,
        data_dir=cfg.data.dir,
        use_ppi=cfg.data.use_ppi_network,
        org_name=cfg.data.org_name,
        significant_ppi=cfg.data.significant_ppi,
    )

    print("\tCreating graph...")
    seed_1 = cfg['model']['seed'][0]
    pyg = _generate_pyg(
        rna_data=rna_df,
        prot_data=prot_df,
        metab_data=metab_df,
        predefined_network=cfg.data.predefined_network,
        output_dir=output_dir,
        graph_name=cfg.data.get("base_network_file", "human_base_network.graphml"),
        random_edges=cfg.data.get("random_edges", False),
        n_random_edges_rna=cfg.data.get("n_random_edges_rna", 0),
        n_random_edges_prot=cfg.data.get("n_random_edges_prot", 0),
        train_test_ratio=cfg.model.train_test_ratio,
        use_causal_edges=cfg.data.use_causal_edges,
        rna_causal_method=cfg.causal.rna_method,
        prot_causal_method=cfg.causal.prot_method,
        seed=seed_1,
        device=cfg.model.get("device", "cpu")
    )

    pyg_copy = copy.deepcopy(pyg)
    if cfg.model.train:
        if cfg.model.train_single_sample:
            print("Starting GNN training on single samples...")
            _trainGNN(
                pyg,
                epochs=cfg.model.epochs,
                weight_path=cfg.model.save_file,
                save_dir=output_dir,
                config=OmegaConf.to_container(cfg=cfg, resolve=True),
            )
        else:
            print("Starting GNN training on combined samples...")
            _trainGNN_combined(
                pyg,
                epochs=cfg.model.epochs,
                weight_path=cfg.model.save_file,
                save_dir=output_dir,
                config=OmegaConf.to_container(cfg=cfg, resolve=True),
            )

    metabs = list(metab_df.columns)[1:]
    prots_l = list(prot_df.columns)[1:]
    rna_l = list(rna_df.columns)[1:]
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(parent_dir, output_dir, "weights", cfg.model.save_file)

    _pathway_analysis(
        pyg_copy,
        weight_path=weight_path,
        metabs=metabs,
        sample_id=cfg.intervention.sample,
        output_dir=output_dir,
    )

    if cfg.intervention.enabled:
        _intervention_analysis(
            pyg=pyg,
            interest_pathway=cfg.intervention.pathway,
            intervention_sample=cfg.intervention.sample,
            intervention_mult=cfg.intervention.mult,
            prots_l=prots_l,
            rna_l=rna_l,
            metabs_l=metabs,
            weight_path=weight_path,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
