import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os

    sys.path.append(os.path.abspath("."))

    import os
    import json
    import numpy as np

    return json, np, os


@app.cell
def _(os):
    logs_files = [x for x in os.listdir("output/logs/") if x.endswith(".json")]
    logs_files.sort()
    logs_files
    return (logs_files,)


@app.cell
def _(json, logs_files):
    file_run_mapping = {}
    for i in logs_files:
        timestamp = i[: i.index("_")]
        print(i)
        with open("output/logs/" + i) as f:
            json_i = json.load(f)
        name_i = json_i["name"]
        print(name_i)
        if file_run_mapping.get(timestamp) is None:
            file_run_mapping[timestamp] = [name_i]
        else:
            file_run_mapping[timestamp].append(name_i)
    return (file_run_mapping,)


@app.cell
def _(file_run_mapping):
    file_run_mapping
    return


@app.cell
def _(file_run_mapping):
    working_timestamps = []
    for _i in file_run_mapping:
        if len(file_run_mapping[_i]) == 5:
            working_timestamps.append(_i)
    return (working_timestamps,)


@app.cell
def _(working_timestamps):
    working_timestamps
    return


@app.cell
def _(np):
    def process_json_loss(inp):
        def stat_loss(arr):
            return np.min(arr), np.mean(arr)

        print(stat_loss(inp["loss_val"]["prot"]))

    return


@app.cell
def _(file_run_mapping, json, np, working_timestamps):
    for _i in working_timestamps:
        prot_train_losses = []
        prot_test_losses = []
        metab_train_losses = []
        metab_test_losses = []
        comb_test_losses = []
        print("Time stamp = ", _i)
        print("Config type = ", file_run_mapping[_i])

        for _j in range(5):
            _file_path = _i + "_run_" + str(_j) + ".json"
            with open("output/logs/" + _file_path) as _f:
                _json_i = json.load(_f)
            # process_json_loss(_json_i)
            prot_train_losses.append(min(_json_i["loss_train"]["prot"]))
            prot_test_losses.append(min(_json_i["loss_val"]["prot"]))

            metab_test_losses.append(min(_json_i["loss_val"]["metab"]))
            metab_train_losses.append(min(_json_i["loss_train"]["metab"]))

            comb_test_losses.append(min(_json_i["loss_val"]["comb"]))

        prot_train_losses = np.array(prot_train_losses)
        prot_test_losses = np.array(prot_test_losses)
        metab_train_losses = np.array(metab_train_losses)
        metab_test_losses = np.array(metab_test_losses)
        comb_test_losses = np.array(comb_test_losses)

        def _print_stats(_t, _arr):
            print(_t, min(_arr), np.mean(_arr), " +- ", np.var(_arr))

        _print_stats("Prot train - ", prot_train_losses)
        _print_stats("Prot test - ", prot_test_losses)
        _print_stats("Metab train - ", metab_train_losses)
        _print_stats("Metab test - ", metab_test_losses)
        _print_stats("Comb test - ", comb_test_losses)
        print(comb_test_losses)
        print("\n\n")
    return


@app.cell
def _():
    len(str(2**30))
    return


@app.cell
def _():
    len(str(2**32))
    return


@app.cell
def _():
    import random

    seeds_test = []
    for _ in range(5):
        seeds_test.append(random.randint(2**30, 2**32 - 1))
    for _i in seeds_test:
        print(_i)
    return


@app.cell
def _():
    return


@app.cell
def _(working_timestamps):
    working_timestamps
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
