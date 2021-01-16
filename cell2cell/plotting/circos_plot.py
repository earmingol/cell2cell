# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc
from matplotlib.path import Path

import networkx as nx
import numpy as np
import pandas as pd

from cell2cell.plotting.aesthetics import get_colors_from_labels, generate_legend


def circos_plot(interaction_space, sender_cells, receiver_cells, ligands, receptors, excluded_score=0, metadata=None,
                sample_col='#SampleID', group_col='Groups', meta_cmap='Set2', cells_cmap='Pastel1', colors=None, ax=None,
                figsize=(10,10), fontsize=14, legend=True, ligand_label_color='dimgray', receptor_label_color='dimgray',
                filename=None):
    '''
    Generates the circos plot in the exact order that sender and receiver cells were provided. Similarly, ligands
    and receptors are sorted by the order they were input.
    '''
    if hasattr(interaction_space, 'interaction_elements'):
        if 'communication_matrix' not in interaction_space.interaction_elements.keys():
            raise ValueError('Run compute_pairwise_communication_scores method before generating circos plots.')

    # Figure setups
    if ax is None:
        R = 1.0
        center = (0, 0)

        fig = plt.figure(figsize=figsize, frameon=False)
        ax = fig.add_axes([0., 0., 1., 1.], aspect='equal')
        ax.set_axis_off()
        ax.set_xlim((-R * 1.05 + center[0]), (R * 1.05 + center[0]))
        ax.set_ylim((-R * 1.05 + center[1]), (R * 1.05 + center[1]))
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = abs(xlim[1] - xlim[0])
        y_range = abs(ylim[1] - ylim[0])

        R = np.nanmin([x_range/2.0, y_range/2.0]) / 1.05
        center = (np.nanmean(xlim), np.nanmean(ylim))

    # Elements to build network
    sorted_nodes = sort_nodes(sender_cells=sender_cells,
                              receiver_cells=receiver_cells,
                              ligands=ligands,
                              receptors=receptors
                              )

    readable_ccc = get_readable_ccc_matrix(interaction_space.interaction_elements['communication_matrix'])

    # Build network
    G = _build_network(sender_cells=sender_cells,
                       receiver_cells=receiver_cells,
                       ligands=ligands,
                       receptors=receptors,
                       sorted_nodes=sorted_nodes,
                       readable_ccc=readable_ccc,
                       excluded_score=excluded_score
                       )

    # Get coordinates
    nodes_dict = get_arc_angles(G=G,
                                sorting_feature='sorting')

    edges_dict = dict()
    for k, v in nodes_dict.items():
        edges_dict[k] = get_cartesian(theta=np.nanmean(v),
                                      radius=0.95*R/2.,
                                      center=center,
                                      angle='degrees')

    small_R = determine_small_radius(edges_dict)

    # Colors
    cells = list(set(sender_cells+receiver_cells))
    if metadata is not None:
        meta = metadata.set_index(sample_col).reindex(cells)
        meta = meta[[group_col]].fillna('NA')
        labels = meta[group_col].unique().tolist()
        if colors is None:
            colors = get_colors_from_labels(labels, cmap=meta_cmap)
        meta['Color'] = meta[group_col].map(colors)
    else:
        meta = pd.DataFrame(index=cells)
    colors = get_colors_from_labels(cells, cmap=cells_cmap)
    meta['Cells-Color'] = meta.index.map(colors)
    # signal_colors = {'ligand' : 'brown', 'receptor' : 'black'}
    signal_colors = {'ligand' : ligand_label_color, 'receptor' : receptor_label_color}

    # Draw edges
    # TODO: Automatically determine lw given the size of the arcs (each ligand or receptor)
    lw = 10
    for l, r in G.edges:
        path = Path([edges_dict[l], center, edges_dict[r]],
                    [Path.MOVETO, Path.CURVE3, Path.CURVE3])


        patch = patches.FancyArrowPatch(path=path,
                                        arrowstyle="->,head_length={},head_width={}".format(lw/3.0,lw/3.0),
                                        lw=lw/2.,
                                        edgecolor='gray', #meta.at[l.split('^')[0], 'Cells-Color'],
                                        zorder=1,
                                        facecolor='none',
                                        alpha=0.15)
        ax.add_patch(patch)

    # Draw nodes
    # TODO: Automatically determine lw given the size of the figure
    cell_legend = dict()
    if metadata is not None:
        meta_legend = dict()
    else:
        meta_legend = None
    for k, v in nodes_dict.items():
        diff = 0.05 * abs(v[1]-v[0]) # Distance to substract and avoid nodes touching each others

        cell = k.split('^')[0]
        cell_color = meta.at[cell, 'Cells-Color']
        cell_legend[cell] = cell_color
        ax.add_patch(Arc(center, R, R,
                         theta1=diff + v[0],
                         theta2=v[1] - diff,
                         edgecolor=cell_color,
                         lw=lw))
        coeff = 1.0
        if metadata is not None:
            coeff = 1.1

            meta_color = meta.at[cell, 'Color']
            ax.add_patch(Arc(center, coeff * R, coeff * R,
                             theta1=diff + v[0],
                             theta2=v[1] - diff,
                             edgecolor=meta_color,
                             lw=10))

            meta_legend[meta.at[cell, group_col]] = meta_color

        label_coord = get_cartesian(theta=np.nanmean(v),
                                    radius=coeff * R/2. + min([small_R*3, coeff * R*0.05]),
                                    center=center,
                                    angle='degrees')

        # Add Labels
        v2 = label_coord
        x = v2[0]
        y = v2[1]
        rotation = np.nanmean(v)

        if x >= 0:
            ha = 'left'
        else:
            ha = 'right'
        va = 'center'

        if (rotation <= 90):
            rotation = abs(rotation)
        elif (rotation <= 180):
            rotation = -abs(rotation - 180)
        elif (rotation <= 270):
            rotation = abs(rotation - 180)
        else:
            rotation = abs(rotation)

        ax.text(x, y,
                k.split('^')[1],
                rotation=rotation,
                rotation_mode="anchor",
                horizontalalignment=ha,
                verticalalignment=va,
                color=signal_colors[G.nodes[k]['signal']],
                fontsize=fontsize
                )

    # Draw legend
    if legend:
        generate_circos_legend(cell_legend=cell_legend,
                               meta_legend=meta_legend,
                               signal_legend=signal_colors,
                               fontsize=fontsize)

    if filename is not None:
        plt.savefig(filename, dpi=300,
                    bbox_inches='tight')
    return ax


def get_arc_angles(G, sorting_feature=None):
    elements = list(set(G.nodes()))
    n_elements = len(elements)

    if sorting_feature is not None:
        sorting_list = [G.nodes[n][sorting_feature] for n in elements]
        elements = [x for _, x in sorted(zip(sorting_list, elements))]

    angles = dict()

    for i, node in enumerate(elements):
        theta1 = (360 / n_elements) * i
        theta2 = (360 / n_elements) * (i + 1)
        angles[node] = (theta1, theta2)
    return angles


def get_cartesian(theta, radius, center=(0,0), angle='radians'):
    if angle == 'degrees':
        theta_ = 2.0*np.pi*theta/360.
    elif angle == 'radians':
        theta_ = theta
    else:
        raise ValueError('Not a valid angle. Use randians or degrees.')
    x = radius*np.cos(theta_) + center[0]
    y = radius*np.sin(theta_) + center[1]
    return (x, y)


def get_readable_ccc_matrix(ccc_matrix):
    readable_ccc = ccc_matrix.copy()
    readable_ccc.index.name = 'LR'
    readable_ccc.reset_index(inplace=True)

    readable_ccc = readable_ccc.melt(id_vars=['LR'])
    readable_ccc['sender'] = readable_ccc['variable'].apply(lambda x: x.split(';')[0])
    readable_ccc['receiver'] = readable_ccc['variable'].apply(lambda x: x.split(';')[1])
    readable_ccc['ligand'] = readable_ccc['LR'].apply(lambda x: x[0])
    readable_ccc['receptor'] = readable_ccc['LR'].apply(lambda x: x[1])
    readable_ccc = readable_ccc[['sender', 'receiver', 'ligand', 'receptor', 'value']]
    readable_ccc.columns = ['sender', 'receiver', 'ligand', 'receptor', 'communication_score']
    return readable_ccc


def sort_nodes(sender_cells, receiver_cells, ligands, receptors):
    sorted_nodes = dict()
    count = 0

    # TODO: Implement a way to put ligands and receptors together for cells included in sender and receiver cells simultaneously
    for c in sender_cells:
        for p in ligands:
            sorted_nodes[(c + '^' + p)] = count
            count += 1

    for c in receiver_cells[::-1]:
        for p in receptors[::-1]:
            sorted_nodes[(c + '^' + p)] = count
            count += 1
    return sorted_nodes


def determine_small_radius(coordinate_dict):
    # Cartesian coordinates
    keys = list(coordinate_dict.keys())
    circle1 = np.asarray(coordinate_dict[keys[0]])
    circle2 = np.asarray(coordinate_dict[keys[1]])
    diff = circle1 - circle2
    distance = np.sqrt(diff[0]**2 + diff[1]**2)
    return distance / 2.0


def _build_network(sender_cells, receiver_cells, ligands, receptors, sorted_nodes, readable_ccc, excluded_score=0):
    G = nx.DiGraph()
    df = readable_ccc.loc[(readable_ccc.sender.isin(sender_cells)) & \
                          (readable_ccc.receiver.isin(receiver_cells)) & \
                          (readable_ccc.ligand.isin(ligands)) & \
                          (readable_ccc.receptor.isin(receptors))
                          ]

    df = df.loc[df['communication_score'] != excluded_score]

    for idx, row in df.iterrows():
        # Generate node names
        node1 = row.sender + '^' + row.ligand
        node2 = row.receiver + '^' + row.receptor

        # Add nodes to the network
        G.add_node(node1,
                   cell=row.sender,
                   protein=row.ligand,
                   signal='ligand',
                   sorting=sorted_nodes[node1])

        G.add_node(node2,
                   cell=row.receiver,
                   protein=row.receptor,
                   signal='receptor',
                   sorting=sorted_nodes[node2])

        # Add edges to. the network
        G.add_edge(node1, node2, weight=row.communication_score)
    return G


def get_node_colors(G, coloring_feature=None, cmap='viridis'):
    if coloring_feature is None:
        raise ValueError('Node feature not specified!')

    # Get what features we have in the network G
    features = set()
    for n in G.nodes():
        features.add(G.nodes[n][coloring_feature])

    # Generate colors for each feature
    NUM_COLORS = len(features)
    cm = plt.get_cmap(cmap)
    feature_colors = dict()
    for i, f in enumerate(features):
        feature_colors[f] = cm(1. * i / NUM_COLORS)  # color will now be an RGBA tuple

    # Map feature colors into nodes
    node_colors = dict()
    for n in G.nodes():
        feature = G.nodes[n][coloring_feature]
        node_colors[n] = feature_colors[feature]
    return node_colors, feature_colors


def generate_circos_legend(cell_legend, signal_legend=None, meta_legend=None, fontsize=14):
    legend_fontsize = int(fontsize * 0.9)

    # Cell legend
    generate_legend(color_dict=cell_legend,
                    loc='center left',
                    bbox_to_anchor=(1.01, 0.5),
                    ncol=1,
                    fancybox=True,
                    shadow=True,
                    title='Cells',
                    fontsize=legend_fontsize
                    )


    if signal_legend is not None:
        # Signal legend
        # generate_legend(color_dict=signal_legend,
        #                 loc='upper center',
        #                 bbox_to_anchor=(0.5, -0.01),
        #                 ncol=2,
        #                 fancybox=True,
        #                 shadow=True,
        #                 title='Signals',
        #                 fontsize=legend_fontsize
        #                 )
        pass

    if meta_legend is not None:
        # Meta legend
        generate_legend(color_dict=meta_legend,
                        loc='center right',
                        bbox_to_anchor=(-0.01, 0.5),
                        ncol=1,
                        fancybox=True,
                        shadow=True,
                        title='Groups',
                        fontsize=legend_fontsize
                        )
