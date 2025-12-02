import polars as pl
import requests
import networkx as nx
from bs4 import BeautifulSoup
import os
from generate_graph import _get_ppi_edges
from utils import _read_file, _save_file


def process_resp(resp):
    resp_i = resp.split("\n")
    dict_entries = {}
    for i in resp_i:
        if len(i) > 0:
            resp_sp_i = i.split("\t")
            gene_id = resp_sp_i[0]
            gene_names = resp_sp_i[1].split(";")[0].split(",")
            gene_names = [x.strip() for x in gene_names]
            dict_entries[gene_id] = gene_names
    return dict_entries


def rename_prot(prot_names):
    url = "https://biodbnet-abcc.ncifcrf.gov/webServices/rest.php/biodbnetRestApi.json?method=db2db&input=kegggeneid&inputValues={}&outputs=genesymbol&taxonId=9606&format=row".format(
        ",".join(prot_names)
    )
    headers = {"User-Agent": "insomnia/10.3.1"}
    response = requests.request("GET", url, headers=headers).json()
    return response


def _generate_base_network(
    rna_df,
    predefined_network,
    graph_name,
    output_dir,
    data_dir,
    use_ppi=False,
    org_name="human",
    significant_ppi=False,
):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), data_dir)
    abs_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), output_dir, "network"
    )
    human_metab_pathway = {}

    if os.path.exists(os.path.join(abs_path, graph_name)) and predefined_network:
        print("\tNetwork already exists...")
        return

    ### Generate Protein-Gene mappings
    prot_df = pl.read_csv(
        os.path.join(data_path, "gene-prot-mapping.tsv"), separator="\t"
    )
    prots = list(prot_df["Entry Name"])
    genes = list(prot_df["From"])
    reviewed = list(prot_df["Reviewed"])
    prot_gene_mapping = {}
    for i in range(len(prots)):
        if reviewed[i] != "reviewed":
            continue
        prot_i = prots[i]
        if type(genes[i]) is float or genes[i] is None:
            continue
        gene_i = genes[i]
        prot_gene_mapping[prot_i] = gene_i

    gene_prot_mapping = {}
    for i in prot_gene_mapping:
        gene_i = prot_gene_mapping[i]
        if type(gene_i) is not list:
            gene_i = list([gene_i])
        if len(gene_i) == 0:
            continue

        for j in gene_i:
            if gene_prot_mapping.get(j) is not None:
                gene_prot_mapping[j].append(i)
            else:
                gene_prot_mapping[j] = [i]

    gene_prot_edges = []
    for i in prot_gene_mapping:
        if type(prot_gene_mapping[i]) is not list:
            prot_gene_mapping[i] = list([prot_gene_mapping[i]])
        for j in prot_gene_mapping[i]:
            gene_prot_edges.append((j, i))
    ###

    ### Generate pathway information file from .xml files downloaded from KEGG
    ### Contains Pathway name, Pathway ID, genes involved, compounds involved
    print("\tRead and collate data from Human Pathway files...")
    pathway_info_dict = {}

    for file_i in os.listdir(os.path.join(data_path, "HumanPathways")):
        file_name = os.path.join(data_path, "HumanPathways", file_i)
        with open(file_name, "r") as f:
            data = f.read()

        Bs_data = BeautifulSoup(data, "xml")
        b_pathway = Bs_data.find("pathway")
        b_unique = Bs_data.find_all("entry", {"type": "gene"})
        pathway_gene_i = set()
        for i in b_unique:
            gene_i = i["name"].split(" ")
            pathway_gene_i.update(gene_i)

        b_unique = Bs_data.find_all("entry", {"type": "compound"})
        pathway_cpd_i = set()
        for i in b_unique:
            cpd_i = i["name"].split(" ")
            pathway_cpd_i.update(cpd_i)
        pathway_cpd_i = list(pathway_cpd_i)

        pathway_info_dict[file_i] = {}
        pathway_info_dict[file_i]["name"] = b_pathway["title"]
        pathway_info_dict[file_i]["gene"] = pathway_gene_i
        pathway_info_dict[file_i]["cpd"] = pathway_cpd_i
    ###

    ### Generate Gene-Protein and Compound-Pathway edges
    add_e = []
    cpd_pathway_e = []
    for i in pathway_info_dict:
        cpd_i = pathway_info_dict[i]["cpd"]
        gene_i = pathway_info_dict[i]["gene"]
        name_i = pathway_info_dict[i]["name"]

        for j in cpd_i:
            for k in gene_i:
                add_e.append([k, j])
            cpd_renamed = j[j.index(":") + 1 :]
            cpd_pathway_e.append([cpd_renamed, name_i])
            if human_metab_pathway.get(cpd_renamed) is None:
                human_metab_pathway[cpd_renamed] = [name_i]
            else:
                human_metab_pathway[cpd_renamed].append(name_i)
    ###
    _save_file(os.path.join(abs_path, "human_metab_pathway_t.pkl"), human_metab_pathway)

    ### Convert the KEGG protein IDs and store
    print("\tConverting KEGG IDs...")
    comb_genes = set()
    for i in pathway_info_dict.keys():
        comb_genes.update(pathway_info_dict[i]["gene"])
    comb_genes = list(comb_genes)

    if not os.path.exists(os.path.join(data_path, "kegg_gene_dict.pkl")):
        step = 100
        store_resps = []
        for i in range(0, len(comb_genes), step):
            print("\tProcessed - ", i, "/", len(comb_genes))
            req_api = "+".join(comb_genes[i : i + step])
            url = "https://rest.kegg.jp/list/" + req_api
            headers = {}
            response = requests.request("POST", url, headers=headers).text
            store_resps.append(response)
        kegg_gene_dict = {}
        for i in store_resps:
            kegg_gene_dict = {**kegg_gene_dict, **process_resp(i)}
        _save_file(os.path.join(data_path, "kegg_gene_dict.pkl"), kegg_gene_dict)
    else:
        print("\tConversion file exists...")
        kegg_gene_dict = _read_file(os.path.join(data_path, "kegg_gene_dict.pkl"))
    ###

    ### Create the Protein-Metabolite edges by renaming the KEGG gene IDs
    print("\tGenerate Protein-Metabolite edges...")
    updated_edges = []
    for i in add_e:
        if i[0].startswith("hsa:"):
            kegg_name = i[0]
            cpd_name = i[1]
            alt_names = kegg_gene_dict[kegg_name]
            for name in alt_names:
                cpd_renamed = cpd_name[cpd_name.index(":") + 1 :]
                if gene_prot_mapping.get(name) is not None:
                    gene_names = gene_prot_mapping[name]
                    for gene in gene_names:
                        updated_edges.append((gene, cpd_renamed))
    ###

    ### Generate PPI edges if needed
    if use_ppi:
        print("\tReading PPI files and adding edges...")
        print("\tSignificant = ", significant_ppi)
        ppi_arr = _get_ppi_edges(org_name=org_name, significant_ppi=significant_ppi)
        rna_l = set(rna_df.columns[1:])
        rr_edges = []
        pp_edges = []
        for i in ppi_arr:
            if i[0] in rna_l and i[1] in rna_l:
                rr_edges.append((i[0], i[1]))
            if (
                gene_prot_mapping.get(i[0]) is not None
                and gene_prot_mapping.get(i[1]) is not None
            ):
                prots_a = gene_prot_mapping[i[0]]
                prots_b = gene_prot_mapping[i[1]]
                for p1 in prots_a:
                    for p2 in prots_b:
                        pp_edges.append((p1, p2))
    ###
    ### Save graph
    print("\tSaving Graph...")
    g_base = nx.Graph()
    g_base.add_edges_from(updated_edges)
    g_base.add_edges_from(gene_prot_edges)
    g_base.add_edges_from(cpd_pathway_e)
    if use_ppi:
        g_base.add_edges_from(rr_edges)
        g_base.add_edges_from(pp_edges)
    nx.write_graphml(g_base, os.path.join(abs_path, graph_name))
    ###
