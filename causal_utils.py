from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod


def run_fci(input_data, cache_path=None, indep_test="kci", bg=None, **kwargs):
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
    return g, labels


def run_pc(input_data, cache_path=None, indep_test="kci", bg=None, **kwargs):
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
    Record = ges(data)

    return Record["G"], labels
