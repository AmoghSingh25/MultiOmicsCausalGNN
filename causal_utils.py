from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod


def run_fci(input_data, cache_path=None, indep_test="gsq", bg=None, **kwargs):
    labels = [f"{col}" for i, col in enumerate(input_data.columns)]
    data = input_data.to_numpy()
    print("\tCache path = ", cache_path)
    g, edges = fci(
        data,
        independence_test_method=indep_test,
        cache_path=cache_path,
        background_knowledge=bg,
        alpha=kwargs["alpha"],
    )
    print(edges)
    print(g[0].graph > 0)
    return g, labels


def run_pc(input_data, cache_path=None, indep_test="gsq", bg=None, **kwargs):
    print("\tCache path = ", cache_path)
    labels = [f"{col}" for i, col in enumerate(input_data.columns)]
    data = input_data.to_numpy().astype("float32")
    cg = pc(
        data,
        cache_path=cache_path,
        indep_test=indep_test,
        background_knowledge=bg,
        alpha=kwargs["alpha"],
    )
    return cg.G, labels


def run_cdnod(input_data, c_idx=1, **kwargs):
    labels = [f"{col}" for i, col in enumerate(input_data.columns)]
    data = input_data.to_numpy()

    cg = cdnod(data, c_idx, alpha=kwargs["alpha"])

    return cg.G, labels


def run_ges(input_data, **kwargs):
    labels = [f"{col}" for i, col in enumerate(input_data.columns)]
    data = input_data.to_numpy()
    Record = ges(data, alpha=kwargs["alpha"])

    return Record["G"], labels


# def remove_duplicate_edges(filename):
#     g = nx.read_graphml("Data/" + filename)
#     g_temp = nx.Graph(g)
#     print("Duplicate edges = ", len(g.edges()) - len(g_temp.edges()))
#     nx.write_graphml(g_temp, "Data/savedNetworks/" + filename)
