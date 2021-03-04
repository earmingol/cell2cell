# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import matplotlib.patches as patches


def get_colors_from_labels(labels, cmap='gist_rainbow', factor=1):
    assert factor >= 1

    colors = dict.fromkeys(labels, ())

    factor = int(factor)
    NUM_COLORS = factor * len(colors)
    cm = plt.get_cmap(cmap)

    for i, label in enumerate(colors.keys()):
        colors[label] = cm((1 + ((factor-1)/factor)) * i / NUM_COLORS)
    return colors


def map_colors_to_metadata(metadata, ref_df=None, colors=None, sample_col='#SampleID', group_col='Groups',
                           cmap='gist_rainbow'):
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
                    title='legend', fontsize=14, sorted_labels=True, fig=None, ax=None):
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