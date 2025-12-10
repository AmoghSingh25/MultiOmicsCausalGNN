# Config description

| Key              | Type    | Default | Description                                      |
|------------------|---------|----|--------------------------------------------------|
| `name` | `string` | `experiment` | Name of the experiment. Used for creating the folder to store results in the output directory and general logging. |
| `description` | `string` | `""` | A brief description of the experiment. |
| `eval` | `bool` | `false` | Flag to indicate if the run is an evaluation run. |
| `debug` | `bool` | `true` | Enables print messages to indicate progress of the runs. |
| **model** | | | Model and training related configuration parameters. |
| > `train` | `bool` | `true` | Indicate if model training needs to be performed. | 
| > `epochs` | `int` | `5` | Number of epochs to train the model for. | 
| > `train_test_ratio` | `int` | `0.7` | Ratio of splitting the data into train:test | 
| > `train_single_sample` | `bool` | `false` | Uses a graph for each sample for training. Is much slower. | 
| > `save_file` | `string` | `saved_weights.pt` | Path of the weights where the model weights need to be saved. | 
| > `runs` | `int` | `1` | Number of training runs to be performed. | 
| > `prot_loss` | `float` | `0` | Influence of protein loss on the combined loss. | 
| > `metab_loss` | `float` | `1` | Influence of protein loss on the combined loss. |
| > `drop_edges` | `float` | `0.5` | Probability of edges to be dropped during training. |
| **data** | | | Data related configuration. |
| > `dir` | `string` | `Data/` | Directory of the input data. |
| > `org_name` | `string` | `human` | Organism of the input data. Requires the pathway files for the organism. |
| > `random_edges` | `bool` | `false` | Add random edges to the network. |
| > `n_random_edges_rna` | `int` | `162` | Number of random RNA-RNA edges to be added. |
| > `n_random_edges_prot` | `int` | `231` | Number of random Protein-Protein edges to be added. |
| > `predefined_network` | `bool` | `false` | Use the pre-computed, stored network or recompute the network. |
| > `use_ppi_network` | `bool` | true | Use PPI edges from String.DB |
| > `use_causal_edges` | `bool` | true | Add in causal discovery edges to the network. |
| > `significant_ppi` | `bool` | true | Use only PPI edges with a high confidence score. |
| > `rna_data` | `string` | `Data/input_data/rna_df.csv` | Location of the input RNA data. |
| > `metab_data` | `string` | `Data/input_data/metab_df.csv` | Location of the input metabolite data. |
| > `prot_data` | `string` | `Data/input_data/prot_df.csv` | Location of the input proteomics data. |
| > `base_network_file` | `string` | `human_base_network.graphml` | Location where the computed network should be stored. |
| **causal** | | | Causal discovery related configuration. |
| > `alpha` | `float` | `0.05` | Alpha value to be used for causal discovery. |
| > `rna_method` | `string` | `ges` | Causal discovery method to use for RNA data. Possible values = `['pc', 'ges', 'fci', 'cdnod', 'llm']` |
| > `prot_method` | `string` | `ges` | Causal discovery method to use for Protein data. Possible values = `['pc', 'ges', 'fci', 'cdnod', 'llm']` |
| > `rna_indep_test` | `string` | `mv_fisherz` | Independence test to be used for causal discovery on RNA data. Possible values = `['fisherz', 'chisq', 'gsq', 'kci', 'mv_fisherz']` |
| > `prot_indep_test` | `string` | `mv_fisherz` | Independence test to be used for causal discovery on Protein data. Possible values = `['fisherz', 'chisq', 'gsq', 'kci', 'mv_fisherz']` |
| > `replaneNanRNA` | `bool` | `false` | Replace NaNs in RNA data with 0 |
| > `replaneNanProt` | `bool` | `true` | Replace NaNs in Protein data with 0 |
| > `rna_keep` | `List` | `None` | RNAs to keep during the causal discovery. | 
| > `prot_keep` | `List` | `None` | Proteins to keep during the causal discovery. |
| > `filt_rna` | `string` | `None` | Location of a .txt file containing list of RNAs to keep. Overrides `rna_keep`. |
| > `filt_prot` | `string` | `None` | Location of a .txt file containing list of Proteins to keep. Overrides `prot_keep`. |
| ***llm*** | | | LLM for causal discovery related configuration |
| >> `objective` | `string` | `None` | Objective of the LLM in the causal discovery. Possible values = `['check', 'orient_check', 'orient', 'None']` |
| >> `model_id` | `string` | `meta-llama-3.1-8b-instruct` | ID of the LLM to be used. Setup to work with LMStudio. |
| >> `temperature` | `float` | `0.7` | Temperature to be used when calling the LLM. |
| >> `rag` | `bool` | `true` | Use RAG for LLM function. |
| **intervention** | | | Post-training intervention analysis |
| > `enabled` | `bool` | `true` | Perform intervention analysis. |
| > `sample` | `int` | `1` | Index of the sample to be used for intervention. |
| > `mult` | `float` | `1.2` | Multiplier used for up-/down-regulate the RNA and Proteins. |
| > `pathway` | `string` | `Central carbon metabolism in cancer` | Pathway used for filtering the RNAs and Proteins to intervene on. |
| **output** | | | Output configuration |
| > `save_dir` | `string` | `output/` | Directory to store the output in. |