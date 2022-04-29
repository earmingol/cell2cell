# Inferring cell-cell interactions from transcriptomes with *cell2cell*
[![PyPI Version][pb]][pypi]
[![Downloads](https://pepy.tech/badge/cell2cell/month)](https://pepy.tech/project/cell2cell)

[pb]: https://badge.fury.io/py/cell2cell.svg
[pypi]: https://pypi.org/project/cell2cell/

## Installation
**First, [install Anaconda following this tutorial](https://docs.anaconda.com/anaconda/install/)**

Once installed, create a new conda environment:
```
conda create -n cell2cell -y python=3.7 jupyter
```

Activate that environment:

```
conda activate cell2cell
```

Then, install cell2cell:
```
pip install cell2cell
```
## Examples

---
![plot](https://github.com/earmingol/cell2cell/blob/master/Logo.png?raw=true)

- A toy example using the **under-the-hood methods of cell2cell** is
  [available here](https://github.com/earmingol/cell2cell/blob/master/examples/cell2cell/Toy-Example.ipynb).
  This case allows personalizing the analyses in a higher level, but it may result **harder to use**.
- A toy example using an Interaction Pipeline for **bulk data** is 
  [available here](https://github.com/earmingol/cell2cell/blob/master/examples/cell2cell/Toy-Example-BulkPipeline.ipynb).
  An Interaction Pipeline makes cell2cell **easier to use**.
- A toy example using an Interaction Pipeline for **single-cell data** is 
  [available here](https://github.com/earmingol/cell2cell/blob/master/examples/cell2cell/Toy-Example-SingleCellPipeline.ipynb).
  An Interaction Pipeline makes cell2cell **easier to use**.  
- An example of using *cell2cell* to infer cell-cell interactions across the **whole
body of *C. elegans*** is [available here](https://github.com/LewisLabUCSD/Celegans-cell2cell)
  
---

![plot](https://github.com/earmingol/cell2cell/blob/master/LogoTensor.png?raw=true)

- Jupyter notebooks for reproducing the results in the manuscript of Tensor-cell2cell
  [are available and can be run online in codeocean.com](https://doi.org/10.24433/CO.0051950.v2).
- **Detailed tutorials for running Tensor-cell2cell and downstream analyses:**
    - [Obtaining patterns of cell-cell communication with Tensor-cell2cell](https://earmingol.github.io/cell2cell/tutorials/ASD/01-Tensor-Factorization-ASD/)
    - [Downstream analysis 1: Factor-specific analyses](https://earmingol.github.io/cell2cell/tutorials/ASD/02-Factor-Specific-ASD/)
    - [Downstream analysis 2: Gene Set Enrichment Analysis](https://earmingol.github.io/cell2cell/tutorials/ASD/03-GSEA-ASD/)
- **Do you have precomputed communication scores?** Re-use them as a prebuilt tensor as [exemplified here](https://github.com/earmingol/cell2cell/blob/master/examples/tensor_cell2cell/Loading-PreBuiltTensor.ipynb).
  This allows reusing previous tensors you built or even plugging in communication scores from other tools.
- **Run Tensor-cell2cell much faster!** An example to perform the analysis using a **NVIDIA GPU** is [available here](https://github.com/earmingol/cell2cell/blob/master/examples/tensor_cell2cell/GPU-Example.ipynb)


---
## Common issues
- When running Tensor-cell2cell (```InteractionTensor.compute_tensor_factorization()``` or ```InteractionTensor.elbow_rank_selection()```), a common error is
associated with Memory. This may happen when the tensor is big enough to make the computer run out of memory when the input of the functions in the parentheses is
  ```init='svd'```. To avoid this issue, just replace it by ```init='random'```.
  
## Ligand-Receptor pairs
- A repository with previously published lists of ligand-receptor pairs [is available here](https://github.com/LewisLabUCSD/Ligand-Receptor-Pairs).
  You can use any of these lists as an input of cell2cell.

## Citation

- **cell2cell** should be cited using this pre-print in bioRxiv:
    - Armingol E., Ghaddar A., Joshi C.J., Baghdassarian H., Shamie I., Chan J.,
      Her H.L., O’Rourke E.J., Lewis N.E. 
      [Inferring a spatial code of cell-cell interactions across a whole animal body](https://www.biorxiv.org/content/10.1101/2020.11.22.392217v3).
       *bioRxiv*, (2020). **DOI: 10.1101/2020.11.22.392217**


- **Tensor-cell2cell** should be cited using this pre-print in bioRxiv:
    - Armingol E., Baghdassarian H., Martino C., Perez-Lopez A., Aamodt C., Knight R., Lewis N.E.
     [Context-aware deconvolution of cell-cell communication with Tensor-cell2cell](https://doi.org/10.1101/2021.09.20.461129)
     *bioRxiv*, (2021). **DOI: 10.1101/2021.09.20.461129**