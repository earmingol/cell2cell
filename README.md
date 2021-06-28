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
- A capsule containing jupyter notebooks for running Tensor-cell2cell (cell2cell v0.4.2) on datasets of **COVID-19 and the embryonic development
  of *C. elegans*** is [available in codeocean.com]()
- **Do you have precomputed communication scores?** Re-use them as a prebuilt tensor as [exemplified here](https://github.com/earmingol/cell2cell/blob/master/examples/tensor_cell2cell/Loading-PreBuiltTensor.ipynb).
  This allows reusing previous tensors you built or even plugging in communication scores from other tools.
- **Run Tensor-cell2cell much faster!** An example to perform the analysis using a **Nvidia GPU** is [available here](https://github.com/earmingol/cell2cell/blob/master/examples/tensor_cell2cell/GPU-Example.ipynb)


---