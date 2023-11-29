# Inferring cell-cell interactions from transcriptomes with *cell2cell*
[![PyPI Version][pb]][pypi]
[![Documentation Status](https://readthedocs.org/projects/cell2cell/badge/?version=latest)](https://cell2cell.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/cell2cell/month)](https://pepy.tech/project/cell2cell)


[pb]: https://badge.fury.io/py/cell2cell.svg
[pypi]: https://pypi.org/project/cell2cell/

## Getting started
For tutorials and documentation, visit [**cell2cell ReadTheDocs**](https://cell2cell.readthedocs.org/) or our [**cell2cell website**](https://earmingol.github.io/cell2cell).



## Installation

<b>Step 1: Install Anaconda</b>
  
First, [install Anaconda following this tutorial](https://docs.anaconda.com/anaconda/install/).


<b>Step 2: Create and Activate a New Conda Environment</b>

```
# Create a new conda environment
conda create -n cell2cell -y python=3.7 jupyter

# Activate the environment
conda activate cell2cell
```


<b>Step 3: Install cell2cell</b>

```
pip install cell2cell
```


## Examples

| cell2cell Examples | Tensor-cell2cell Examples |
| --- | --- |
| ![cell2cell Logo](https://github.com/earmingol/cell2cell/blob/master/Logo.png?raw=true) | ![Tensor-cell2cell Logo](https://github.com/earmingol/cell2cell/blob/master/LogoTensor.png?raw=true) |
| - [Step-by-step Pipeline](https://github.com/earmingol/cell2cell/blob/master/examples/cell2cell/Toy-Example.ipynb)<br>- [Interaction Pipeline for Bulk Data](https://earmingol.github.io/cell2cell/tutorials/Toy-Example-BulkPipeline)<br>- [Interaction Pipeline for Single-Cell Data](https://earmingol.github.io/cell2cell/tutorials/Toy-Example-SingleCellPipeline)<br>- [Whole Body of *C. elegans*](https://github.com/LewisLabUCSD/Celegans-cell2cell) | - [Obtaining patterns of cell-cell communication](https://earmingol.github.io/cell2cell/tutorials/ASD/01-Tensor-Factorization-ASD/)<br>- [Downstream 1: Factor-specific analyses](https://earmingol.github.io/cell2cell/tutorials/ASD/02-Factor-Specific-ASD/)<br>- [Downstream 2: Patterns to functions (GSEA)](https://earmingol.github.io/cell2cell/tutorials/ASD/03-GSEA-ASD/)<br>- [Tensor-cell2cell in Google Colab (**GPU**)](https://colab.research.google.com/drive/1T6MUoxafTHYhjvenDbEtQoveIlHT2U6_?usp=sharing)<br>- [Communication patterns in **Spatial Transcriptomics**](https://earmingol.github.io/cell2cell/tutorials/Tensor-cell2cell-Spatial/) |

Reproducible runs of the analyses in the [Tensor-cell2cell paper](https://doi.org/10.1038/s41467-022-31369-2) are available at [CodeOcean.com](https://doi.org/10.24433/CO.0051950.v2)

## LIANA & Tensor-cell2cell

Explore our tutorials for using Tensor-cell2cell with [LIANA](https://github.com/saezlab/liana-py) at [ccc-protocols.readthedocs.io](https://ccc-protocols.readthedocs.io/).

## Common Issues

- **Memory Errors with Tensor-cell2cell:** If you encounter memory errors when performing tensor factorizations, try replacing `init='svd'` with `init='random'`.
  
## Ligand-Receptor Pairs
Find a curated list of ligand-receptor pairs for your analyses at our [GitHub Repository](https://github.com/LewisLabUCSD/Ligand-Receptor-Pairs).

## Citation

Please cite our work using the following references:

- **cell2cell**: [Inferring a spatial code of cell-cell interactions across a whole animal body](https://doi.org/10.1371/journal.pcbi.1010715).
  *PLOS Computational Biology, 2022*

- **Tensor-cell2cell**: [Context-aware deconvolution of cell-cell communication with Tensor-cell2cell](https://doi.org/10.1038/s41467-022-31369-2).
  *Nature Communications, 2022.*

- **LIANA & Tensor-cell2cell tutorials**: [Combining LIANA and Tensor-cell2cell to decipher cell-cell communication across multiple samples](https://doi.org/10.1101/2023.04.28.538731).
  *bioRxiv, 2023*