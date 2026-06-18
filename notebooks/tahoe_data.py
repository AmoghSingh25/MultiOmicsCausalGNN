import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from datasets import load_dataset
    return (load_dataset,)


@app.cell
def _(load_dataset):
    ds = load_dataset("tahoebio/Tahoe-100m", streaming=True, split='train')
    return (ds,)


@app.cell
def _(ds):
    next(ds.iter(3))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
