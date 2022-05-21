from __future__ import absolute_import

from cell2cell.plotting.aesthetics import get_colors_from_labels, map_colors_to_metadata, generate_legend
from cell2cell.plotting.ccc_plot import clustermap_ccc
from cell2cell.plotting.cci_plot import clustermap_cci
from cell2cell.plotting.circular_plot import circos_plot
from cell2cell.plotting.pval_plot import dot_plot
from cell2cell.plotting.factor_plot import context_boxplot, loading_clustermap, ccc_networks_plot
from cell2cell.plotting.pcoa_plot import pcoa_3dplot
from cell2cell.plotting.tensor_plot import tensor_factors_plot, tensor_factors_plot_from_loadings
from cell2cell.plotting.umap_plot import umap_biplot