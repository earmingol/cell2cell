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