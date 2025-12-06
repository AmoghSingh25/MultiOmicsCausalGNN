import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from utils import _read_file
    metric = _read_file("output/comp_metrics.pkl")
    return (metric,)


@app.cell
def _(metric):
    metric
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
