import warnings
from tqdm import tqdm
import os
from MultiLayerHuman import MultiLayerHuman
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from utils import _read_file

warnings.filterwarnings("ignore")


def get_top_pathways(p_s, p_n):
    s, n = zip(*sorted(zip(p_s, p_n), reverse=False))
    return list(s), list(n)


def get_pathway_mean_sum(metab_vals, metabs):

    abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data")

    metab_pathways = _read_file(os.path.join(abs_path, "human_metab_pathway_t.pkl"))

    rev_metab_pathway = {}
    for i in metab_pathways:
        pt_i = metab_pathways[i]
        for j in pt_i:
            if rev_metab_pathway.get(j) is not None:
                rev_metab_pathway[j].append(i)
            else:
                rev_metab_pathway[j] = [i]

    metab_vals = torch.log(
        metab_vals
    )  # Log transformation before computing pathway scores

    metab_pathways_l = set()
    for i in metab_pathways.values():
        metab_pathways_l.update(i)
    metab_pathways_l = {k: {"vals": []} for k in metab_pathways_l}

    for i in range(len(metab_vals)):
        metab_name = metabs[i]
        if metab_pathways.get(metab_name) is None:
            continue
        for j in metab_pathways[metab_name]:
            metab_pathways_l[j]["vals"].append(metab_vals[i].item())
    pathway_names = []
    activity_scores = []

    for i in metab_pathways_l:
        val_i = np.array(metab_pathways_l[i]["vals"])
        k = val_i.shape[0]
        N = len(rev_metab_pathway[i])
        metab_pathways_l[i]["activity_score"] = np.sum(val_i) / (k / N)
        activity_scores.append(metab_pathways_l[i]["activity_score"])
        pathway_names.append(i)

    activity_scores = np.nan_to_num(np.array(activity_scores))

    ## Activity scores
    scores, names = get_top_pathways(activity_scores, pathway_names)

    comp_vals = [-scores[names.index(i)] for i in names]

    return comp_vals, names


def _pathway_analysis(pyg, weight_path, metabs, sample_id, output_dir):
    print("Performing pathway analysis...")
    device = torch.device("cpu")
    model = MultiLayerHuman(inp_dim=pyg["protein"].x.shape[1]).to(device)

    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        out = model(pyg.x_dict, pyg.edge_index_dict)

    comp_vals, names = get_pathway_mean_sum(out["metabolite"][:, sample_id], metabs)

    abs_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), output_dir, "figures"
    )

    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    print("\tPathway analysis graphs saved to ", abs_path)
    _save_pathway_fig(names, comp_vals, os.path.join(abs_path, "activity_scores.png"))


def _intervention_analysis(
    pyg,
    interest_pathway,
    intervention_sample,
    intervention_mult,
    prots_l,
    rna_l,
    metabs_l,
    weight_path,
    output_dir,
):
    print("Performing intervention analysis...")
    metab_pathway = set()
    abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Data")

    print("\tReading network file ...")
    metab_pathways = _read_file(os.path.join(abs_path, "human_metab_pathway_t.pkl"))

    g = nx.read_graphml(os.path.join(abs_path, "human_base_network.graphml"))

    for i in metab_pathways:
        if interest_pathway in metab_pathways[i]:
            metab_pathway.add(i)
    metab_pathway = list(metab_pathway)
    prot_interest = set()
    prot_indices = set()

    print("\tIdentifying connected proteins ...")

    for i in tqdm(g.edges):
        if i[0] in metab_pathway and i[1] in prots_l:
            prot_interest.add(i[1])
            prot_indices.add(prots_l.index(i[1]))
        elif i[1] in metab_pathway and i[0] in prots_l:
            prot_interest.add(i[0])
            prot_indices.add(prots_l.index(i[0]))

    print("\tIdentifying connected RNAs ...")
    prot_interest = list(prot_interest)
    prot_indices = list(prot_indices)
    rna_indices = set()
    rna_interest = set()
    rna_s = {rna_l[x]: x for x in range(len(rna_l))}
    for i in tqdm(prot_interest):
        prot_i_edges = list(g.edges(i))
        for e in prot_i_edges:
            if e[0] in rna_s:
                rna_interest.add(e[0])
                rna_indices.add(rna_s[e[0]])
            elif e[1] in rna_s:
                rna_interest.add(e[1])
                rna_indices.add(rna_s[e[1]])
    del rna_s

    device = torch.device("cpu")
    model = MultiLayerHuman(inp_dim=pyg["protein"].x.shape[1]).to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        out = model(pyg.x_dict, pyg.edge_index_dict)

    metab_init = out["metabolite"][:, intervention_sample]
    prot_init = out["protein"][:, intervention_sample]

    for i in prot_indices:
        pyg["protein"].x[i, intervention_sample] = (
            pyg["protein"].x[i, intervention_sample] * intervention_mult
        )

    for i in rna_indices:
        pyg["rna"].x[i, intervention_sample] = (
            pyg["rna"].x[i, intervention_sample] * intervention_mult
        )

    with torch.no_grad():
        out = model(pyg.x_dict, pyg.edge_index_dict)

    protein_new = out["protein"][:, intervention_sample]
    diff = np.array(protein_new - prot_init)
    temp_diff = np.abs(np.array(diff))
    temp_diff.sort()
    filt_val = temp_diff[-10]
    condition = np.abs(diff) >= filt_val
    short_prots = np.array(prots_l)[condition]
    diff_prots = diff[condition]
    for i in range(len(short_prots)):
        if ";" in short_prots[i]:
            short_prots[i] = short_prots[i].split(";")[0]

    fig_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), output_dir, "figures"
    )
    _save_change_fig(
        diff_prots, short_prots, os.path.join(fig_path, "protein_change_human.png")
    )

    metab_new = out["metabolite"][:, intervention_sample]
    diff = np.array(metab_new - metab_init)
    temp_diff = np.abs(np.array(diff))
    temp_diff.sort()
    filt_val = temp_diff[-10]
    condition = np.abs(diff) >= filt_val
    short_metabs = np.array(metabs_l)[condition]
    diff_metab = diff[condition]
    _save_change_fig(
        diff_metab,
        short_metabs,
        os.path.join(fig_path, "metabolite_change_human.png"),
        omic="m",
    )

    l1, l2 = get_pathway_mean_sum(metab_init, metabs_l)
    l1l, l2l = get_pathway_mean_sum(metab_new, metabs_l)

    pathway_change = {}
    for i in range(len(l2)):
        pathway_change[l2[i]] = (l1[i], l1l[l2l.index(l2[i])])
    comp_vals_sum = [l1l[l2l.index(i)] for i in l2]

    _save_pathway_diff_fig(
        l1, comp_vals_sum, l2, os.path.join(fig_path, "pathway_diff_human.png")
    )

    rel_as_names, rel_as_scores = get_pathway_mean_sum(metab_new / metab_init, metabs_l)
    _save_pathway_fig(
        rel_as_scores,
        rel_as_names,
        os.path.join(fig_path, "rel_activity_post_intervention.png"),
        plot_type="rel",
    )
    print("\tIntervention analysis plots saved")


def _save_pathway_diff_fig(bef_val, aft_val, pathways, path, t="total"):
    plt.figure(dpi=600)
    fig, ax = plt.subplots()
    y = np.arange(10)
    fig.set_figheight(10)
    fig.set_figwidth(18)
    ret = ax.barh(
        y + 0.2,
        bef_val[:10],
        0.4,
        label="Before Intervention",
        color="g" if t == "normalised" else "b",
    )
    ret2 = ax.barh(
        y - 0.2,
        aft_val[:10],
        0.4,
        label="After Intervention",
        color="y" if t == "normalised" else "salmon",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(pathways[:10], fontsize=22)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)

    ax.set_ylabel("Pathway", fontsize=22)
    ax.set_xlabel("Pathway activity", fontsize=22)

    ax.bar_label(ret, fmt="{:,.2f}", fontsize=18)
    ax.bar_label(ret2, fmt="{:,.2f}", fontsize=18)
    ax.set_title(
        f"Comparison of {t} pathway activities with RNA upregulation",
        fontsize=24,
        pad=20,
    )
    plt.legend(fontsize=18)
    plt.savefig(path, dpi=600, bbox_inches="tight")


def _save_change_fig(vals, names, path, omic="p"):
    plt.figure(dpi=600)
    x = np.arange(len(vals))

    fig, ax = plt.subplots(figsize=(18, 10))
    ret = ax.bar(x, vals, 0.5, label="", color="navy" if omic == "p" else "salmon")

    ax.set_ylabel(
        f"Change in {'protein' if omic == 'p' else 'metabolite'} level",
        fontsize=22,
        labelpad=20,
    )
    ax.set_xlabel(
        f"{'Proteins' if omic == 'p' else 'Metabolites'}", fontsize=22, labelpad=20
    )
    ax.set_title(
        f"{'Protein' if omic == 'p' else 'Metabolite'} changes after upregulating RNA",
        fontsize=22,
        pad=20,
    )
    ax.set_xticks([i + 0.15 for i in x])
    ax.set_xticklabels(names, rotation=90, ha="right", fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    ax.bar_label(ret, labels=[f"{val.get_height():.4f}" for val in ret])
    plt.tight_layout()
    plt.savefig(path, dpi=600)


def _save_pathway_fig(vals, names, path, plot_type="abs"):
    plt.figure(dpi=600)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ret = ax.barh(vals[:10], names[:10], color="b")

    ax.set_ylabel("Pathway", fontsize=22)
    ax.set_xlabel("Pathway activity", fontsize=22)

    ax.set_xticklabels(ax.get_xticklabels(), ha="right", fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

    ax.bar_label(ret, fmt="{:,.2f}" if plot_type == "abs" else "{:,.4f}")
    ax.set_title(
        "Top 10 pathways by Activity Scores"
        if plot_type == "abs"
        else "Top 10 pathways by relative Activity Score post-intervention",
        fontsize=22,
    )
    plt.savefig(path, dpi=600, bbox_inches="tight")
