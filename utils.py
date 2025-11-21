import pickle


def _save_file(path, var):
    with open(path, "wb") as file:
        pickle.dump(var, file)


def _read_file(path):
    with open(path, "rb") as file:
        var = pickle.load(file)
    return var
