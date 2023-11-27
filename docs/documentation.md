# Documentation for *cell2cell*

This documentation is for our *cell2cell* suite, which includes the [regular cell2cell](https://doi.org/10.1371/journal.pcbi.1010715)
and [Tensor-cell2cell](https://doi.org/10.1038/s41467-022-31369-2) tools. The former is for inferring cell-cell interactions
and communication in one sample or context, while the latter is for deconvolving complex patterns
of cell-cell communication across multiple samples or contexts simultaneously into interpretable factors
representing patterns of communication.

Here, multiple classes and functions are implemented to facilitate the analyses, including a variety of
visualizations to simplify the interpretation of results:

- **cell2cell.analysis** : Includes simplified pipelines for running the analyses, and functions for downstream analyses of Tensor-cell2cell
- **cell2cell.clustering** : Includes multiple scipy-based functions for performing clustering methods.
- **cell2cell.core** : Includes the core functions for inferring cell-cell interactions and communication. It includes scoring methods, cell classes, and interaction spaces.
- **cell2cell.datasets** : Includes toy datasets and annotations for testing functions in basic scenarios.
- **cell2cell.external** : Includes built-in approaches borrowed from other tools to avoid incompatibilities (e.g. UMAP, tensorly, and PCoA).
- **cell2cell.io** : Includes functions for opening and saving diverse types of files.
- **cell2cell.plotting** : Includes all the visualization options that *cell2cell* offers.
- **cell2cell.preprocessing** : Includes functions for manipulating data and variables (e.g. data preprocessing, integration, permutation, among others).
- **cell2cell.spatial** : Includes filtering of cell-cell interactions results given intercellular distance, as well as defining neighborhoods by grids or moving windows.
- **cell2cell.stats** : Includes statistical analyses such as enrichment analysis, multiple test correction methods, permutation approaches, and Gini coefficient.
- **cell2cell.tensor** : Includes all functions pertinent to the analysis of *Tensor-cell2cell*
- **cell2cell.utils** : Includes general utilities for analyzing networks and performing parallel computing.


Below, all the inputs, parameters (including their different options), and outputs are detailed. Source code of the functions is also included.


::: cell2cell.analysis
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.clustering
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.core
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.datasets
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.external
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.io
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.plotting
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.preprocessing
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.spatial
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.stats
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.tensor
    handler: python
    rendering:
      show_root_heading: true
      show_source: true

::: cell2cell.utils
    handler: python
    rendering:
      show_root_heading: true
      show_source: true
