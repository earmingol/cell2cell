# Inferring cell-cell interactions from transcriptomes with *cell2cell*
[![PyPI Version][pb]][pypi]
[![PyPI Downloads][db]][pypi]

[pb]: https://badge.fury.io/py/cell2cell.svg
[pypi]: https://pypi.org/project/cell2cell/
[db]: https://img.shields.io/pypi/dm/cell2cell?label=pypi%20downloads

## Installation
Create a new conda environment:
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
- A capsule containing jupyter notebooks for reproducing the results in the manuscript of Tensor-cell2cell  is [available in codeocean.com](https://doi.org/10.24433/CO.0051950.v1).
  It specifically contains analyses on datasets of **COVID-19 and the embryonic development
  of *C. elegans***, such as the evaluation of changes in
  cell-cell communication dependent on [different severities of COVID-19](https://files.codeocean.com/files/verified/bffc457e-caa6-4c39-b869-f52330804db0_v1.0/results.5f6a2e3f-70d2-4547-a1c2-7304335758d9/06-BALF-Tensor-Factorization.html),
  and changes dependent on [multiple time points of the *C. elegans* development](https://files.codeocean.com/files/verified/bffc457e-caa6-4c39-b869-f52330804db0_v1.0/results.5f6a2e3f-70d2-4547-a1c2-7304335758d9/08-Celegans-Tensor-Factorization.html)
- **Do you have precomputed communication scores?** Re-use them as a prebuilt tensor as [exemplified here](https://github.com/earmingol/cell2cell/blob/master/examples/tensor_cell2cell/Loading-PreBuiltTensor.ipynb).
  This allows reusing previous tensors you built or even plugging in communication scores from other tools.
- **Run Tensor-cell2cell much faster!** An example to perform the analysis using a **Nvidia GPU** is [available here](https://github.com/earmingol/cell2cell/blob/master/examples/tensor_cell2cell/GPU-Example.ipynb)


---
## Common issues
- When running Tensor-cell2cell (```InteractionTensor.compute_tensor_factorization()``` or ```InteractionTensor.elbow_rank_selection()```), a common error is
associated with Memory. This may happen when the tensor is big enough to make the computer run out of memory when the input of the functions in the parentheses is
  ```init='svd'```. To avoid this issue, just replace it by ```init='random'```.
  
## Ligand-Receptor pairs
- A repository with previously published lists of ligand-receptor pairs [is available here](https://github.com/LewisLabUCSD/Ligand-Receptor-Pairs).
  You can use any of these lists as an input of cell2cell.