## Downloading Pathway Data

Use RStudio to download the EnrichmentBrowser package and download the KEGG pathways for other organisms

```
library("EnrichmentBrowser")
sarpathway <- downloadPathways("hsa", out.dir="Data/HumanPathways", cache=FALSE)
```

Requires Pathway files and PPI network files for generation of the graph links (RNA-Protein-Metabolite-Pathway).
Currently supplied HumanPathway files from KEGG and PPI networks of Human and Mouse from STRING.DB

RNA data should use Official Gene Symbols as columns
Protein data should use UniProt Protein IDs as columns
Metabolite data should use KEGG IDs as columns

config.data.dir + "gene-prot-mapping.tsv" - Contains the gene and the protein they encode downloaded from UniProt ID Mapping

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager before running the setup

```bash
# Clone repository
git clone https://github.com/AmoghSingh25/GeneProtSim.git

# Create venv and download packages
uv sync
```

## Running the code

A custom configuration can be specified in `configs/` by using the `base.yaml` as the default config and modifying the required values. The config can be similar to `configs/gliob.yaml`, modifying the parameters required.

```bash
uv run causal_disc.py # If causal edges are to be used
#OR
uv run causal_disc.py -cn custom_config # Specify a custom config to use for running causal discovery

uv run main.py
#OR
uv run main.py -cn custom_config # Specify a custom config
```

`causal_disc.py` must be run before running `main.py` if causal links need to be computed and added to the network. The corresponding changes must also be made in the config file (Enabling causal edges, specifying causal discovery parameters).

## [Configuration description](docs/config_desc.md)
[Config description](docs/config_desc.md)

## Working
- Gene expression and proteomics filtered using https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp
- Wikipages taken from wikimedia export, split into docs using WikiExtractor and loaded into RAG.

## Improvements

- Allow for a larger set of IDs as columns and conversion of IDs