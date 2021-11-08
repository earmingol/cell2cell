# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np


def get_colors_from_labels(labels, cmap='gist_rainbow', factor=1):
    '''Generates colors for each label in a list given a colormap

    Parameters
    ----------
    labels : list
        A list of labels to assign a color.

    cmap : str, default='gist_rainbow'
        A matplotlib color palette name.

    factor : int, default=1
        Factor to amplify the separation of colors.

    Returns
    -------
    colors : dict
        A dictionary where the keys are the labels and the values
        correspond to the assigned colors.
    '''
    assert factor >= 1

    colors = dict.fromkeys(labels, ())

    factor = int(factor)
    cm_ = plt.get_cmap(cmap)

    is_number = all((isinstance(e, float) or isinstance(e, int)) for e in labels)

    if not is_number:
        NUM_COLORS = factor * len(colors)
        for i, label in enumerate(colors.keys()):
            colors[label] = cm_((1 + ((factor-1)/factor)) * i / NUM_COLORS)
    else:
        max_ = np.nanmax(labels)
        min_ = np.nanmin(labels)
        norm = Normalize(vmin=-min_, vmax=max_)

        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for label in colors.keys():
            colors[label] = m.to_rgba(label)
    return colors


def map_colors_to_metadata(metadata, ref_df=None, colors=None, sample_col='#SampleID', group_col='Groups',
                           cmap='gist_rainbow'):
    '''Assigns a color to elements in a dataframe containing metadata.

    Parameters
    ----------
    metadata : pandas.DataFrame
        A dataframe with metadata for specific elements.

    ref_df : pandas.DataFrame
        A dataframe whose columns contains a subset of
        elements in the metadata.

    colors : dict, default=None
        Dictionary containing tuples in the RGBA format for indicating colors
        of major groups of cells. If colors is specified, cmap will be
        ignored.

    sample_col : str, default='#SampleID'
        Column in the metadata for elements to color.

    group_col : str, default='Groups'
        Column in the metadata containing the major groups of the elements
        to color.

    cmap : str, default='gist_rainbow'
        Name of the color palette for coloring the major groups of elements.

    Returns
    -------
    new_colors : pandas.DataFrame
        A pandas dataframe where the index is the list of elements in the
        sample_col and the column group_col contains the colors assigned
        to each element given their groups.
    '''
    if ref_df is not None:
        meta_ = metadata.set_index(sample_col).reindex(ref_df.columns)
    else:
        meta_ = metadata.set_index(sample_col)
    labels = meta_[group_col].unique().tolist()
    if colors is None:
        colors = get_colors_from_labels(labels, cmap=cmap)
    else:
        upd_dict = dict([(v, (1., 1., 1., 1.)) for v in labels if v not in colors.keys()])
        colors.update(upd_dict)

    new_colors = meta_[group_col].map(colors)
    new_colors.index = meta_.index
    new_colors.name = group_col.capitalize()

    return new_colors


def generate_legend(color_dict, loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1, fancybox=True, shadow=True,
                    title='Legend', fontsize=14, sorted_labels=True, fig=None, ax=None):
    '''Adds a legend to a previous plot or displays an independent legend
    given specific colors for labels.

    Parameters
    ----------
    color_dict : dict
        Dictionary containing tuples in the RGBA format for indicating colors
        of major groups of cells. Keys are the labels and values are the RGBA
        tuples.

    loc : str, default='center left'
        Alignment of the legend given the location specieid in bbox_to_anchor.

    bbox_to_anchor : tuple, default=(1.01, 0.5)
        Location of the legend in a (X, Y) format. For example, if you want
        your axes legend located at the figure's top right-hand corner instead
        of the axes' corner, simply specify the corner's location and the
        coordinate system of that location, which in this case would be (1, 1).

    ncol : int, default=1
        Number of columns to display the legend.

    fancybox : boolean, default=True
        Whether round edges should be enabled around the FancyBboxPatch which
        makes up the legend's background.

    shadow : boolean, default=True
        Whether to draw a shadow behind the legend.

    title : str, default='Legend'
        Title of the legend box

    fontsize : int, default=14
        Size of the text in the legends.

    sorted_labels : boolean, default=True
        Whether alphabetically sorting the labels.

    fig : matplotlib.figure.Figure, default=None
        Figure object to add a legend. If fig=None and ax=None, a new empty
        figure will be generated.

    ax : matplotlib.axes.Axes, default=None
        Axes instance for a plot.

    Returns
    -------
    legend1 : matplotlib.legend.Legend
        A legend object in a figure.
    '''
    color_patches = []
    if sorted_labels:
        iteritems = sorted(color_dict.items())
    else:
        iteritems = color_dict.items()
    for k, v in iteritems:
        color_patches.append(patches.Patch(color=v, label=str(k).replace('_', ' ')))

    if ax is None:
        legend1 = plt.legend(handles=color_patches,
                             loc=loc,
                             bbox_to_anchor=bbox_to_anchor,
                             ncol=ncol,
                             fancybox=fancybox,
                             shadow=shadow,
                             title=title,
                             fontsize=fontsize)
    else:
        legend1 = ax.legend(handles=color_patches,
                             loc=loc,
                             bbox_to_anchor=bbox_to_anchor,
                             ncol=ncol,
                             fancybox=fancybox,
                             shadow=shadow,
                             title=title,
                             fontsize=fontsize)

    if (fig is None) & (ax is None):
        plt.setp(legend1.get_title(), fontsize=fontsize)
        plt.gca().add_artist(legend1)
    else:
        plt.setp(legend1.get_title(), fontsize=fontsize)
    return legend1