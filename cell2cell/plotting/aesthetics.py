# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt


def get_colors_from_labels(labels, cmap='gist_rainbow', factor=1):
    assert factor >= 1

    colors = dict.fromkeys(labels, ())

    factor = int(factor)
    NUM_COLORS = factor * len(colors)
    cm = plt.get_cmap(cmap)

    for i, label in enumerate(colors.keys()):
        colors[label] = cm((1 + ((factor-1)/factor)) * i / NUM_COLORS)
    return colors


def map_colors_to_metadata(df, metadata, colors=None, sample_col='#SampleID', group_col='Groups',
                           meta_cmap='gist_rainbow'):
    meta_ = metadata.set_index(sample_col).reindex(df.columns)
    labels = meta_[group_col].unique().tolist()
    if colors is None:
        colors = get_colors_from_labels(labels, cmap=meta_cmap)
    else:
        upd_dict = dict([(v, (1., 1., 1., 1.)) for v in labels if v not in colors.keys()])
        colors.update(upd_dict)

    new_colors = meta_[group_col].map(colors)
    new_colors.index = meta_.index
    new_colors.name = group_col.capitalize()

    return new_colors