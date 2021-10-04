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
    '''Generates the circos plot in the exact order that sender and
    receiver cells are provided. Similarly, ligands and receptors are
    sorted by the order they are input.

    Parameters
    ----------
    interaction_space : cell2cell.core.interaction_space.InteractionSpace
        Interaction space that contains all a distance matrix after running the
        the method compute_pairwise_communication_scores. Alternatively, this
        object can a SingleCellInteractions or a BulkInteractions object after
        running the method compute_pairwise_communication_scores.

    sender_cells : list
        List of cells to be included as senders.

    receiver_cells : list
        List of cells to be included as receivers.

    ligands : list
        List of genes/proteins to be included as ligands produced by the
        sender cells.

    receptors : list
        List of genes/proteins to be included as receptors produced by the
        receiver cells.

    excluded_score : float, default=0
        Rows that have a communication score equal or lower to this will
        be dropped from the network.

    metadata : pandas.Dataframe, default=None
        Metadata associated with the cells, cell types or samples in the
        matrix containing CCC scores. If None, cells will be color only by
        individual cells.

    sample_col : str, default='#SampleID'
        Column in the metadata for the cells, cell types or samples
        in the matrix containing CCC scores.

    group_col : str, default='Groups'
        Column in the metadata containing the major groups of cells, cell types
        or samples in the matrix with CCC scores.

    meta_cmap : str, default='Set2'
        Name of the matplotlib color palette for coloring the major groups
        of cells.

    cells_cmap : str, default='Pastel1'
        Name of the color palette for coloring individual cells.

    colors : dict, default=None
        Dictionary containing tuples in the RGBA format for indicating colors
        of major groups of cells. If colors is specified, meta_cmap will be
        ignored.

    ax : matplotlib.axes.Axes, default=None
        Axes instance for a plot.

    figsize : tuple, default=(10, 10)
        Size of the figure (width*height), each in inches.

    fontsize : int, default=14
        Font size for ligand and receptor labels.

    legend : boolean, default=True
        Whether including legends for cell and cell group colors as well
        as ligand/receptor colors.

    ligand_label_color : str, default='dimgray'
        Name of the matplotlib color palette for coloring the labels of
        ligands.

    receptor_label_color : str, default='dimgray'
        Name of the matplotlib color palette for coloring the labels of
        receptors.

    filename : str, default=None
        Path to save the figure of the elbow analysis. If None, the figure is not
        saved.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes instance containing a circos plot.
    '''
    if hasattr(interaction_space, 'interaction_elements'):
        if 'communication_matrix' not in interaction_space.interaction_elements.keys():
            raise ValueError('Run the method compute_pairwise_communication_scores() before generating circos plots.')
        else:
            readable_ccc = get_readable_ccc_matrix(interaction_space.interaction_elements['communication_matrix'])
    elif hasattr(interaction_space, 'interaction_space'):
        if 'communication_matrix' not in interaction_space.interaction_space.interaction_elements.keys():
            raise ValueError('Run the method compute_pairwise_communication_scores() before generating circos plots.')
        else:
            readable_ccc = get_readable_ccc_matrix(interaction_space.interaction_space.interaction_elements['communication_matrix'])
    else:
        raise ValueError('Not a valid interaction_space')

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
    # TODO: Add option to select sort_by: None (as input), cells, proteins or metadata
    sorted_nodes = sort_nodes(sender_cells=sender_cells,
                              receiver_cells=receiver_cells,
                              ligands=ligands,
                              receptors=receptors
                              )

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
        meta['Color'] = [colors[idx] for idx in meta[group_col]]
    else:
        meta = pd.DataFrame(index=cells)
    # Colors for cells, not major groups
    colors = get_colors_from_labels(cells, cmap=cells_cmap)
    meta['Cells-Color'] = [colors[idx] for idx in meta.index]
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
    '''Obtains the angles of polar coordinates to plot nodes as arcs of
     a circumference.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        A networkx graph.

    sorting_feature : str, default=None
        A node attribute present in the dictionary associated with
        each node. The values associated with this attributed will
        be used for sorting the nodes.

    Returns
    -------
    angles : dict
        A dictionary containing the angles for positioning the
        nodes in polar coordinates. Keys are the node names and
        values are tuples with angles for the start and end of
        the arc that represents a node.
    '''
    elements = list(set(G.nodes()))
    n_elements = len(elements)

    if sorting_feature is not None:
        sorting_list = [G.nodes[n][sorting_feature] for n in elements]
        elements = [x for _, x in sorted(zip(sorting_list, elements), key=lambda pair: pair[0])]

    angles = dict()

    for i, node in enumerate(elements):
        theta1 = (360 / n_elements) * i
        theta2 = (360 / n_elements) * (i + 1)
        angles[node] = (theta1, theta2)
    return angles


def get_cartesian(theta, radius, center=(0,0), angle='radians'):
    '''Performs a polar to cartesian coordinates conversion.

    Parameters
    ----------
    theta : float or ndarray
        An angle for a polar coordinate.

    radius : float
        The radius in a polar coordinate.

    center : tuple, default=(0,0)
        The center of the circle in the cartesian coordinates.

    angle : str, default='radians'
        Type of angle that theta is. Options are:
         - 'degrees' : from 0 to 360
         - 'radians' : from 0 to 2*numpy.pi

    Returns
    -------
    (x, y) : tuple
        Cartesian coordinates for X and Y axis respective.
    '''
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
    '''Transforms a CCC matrix from an InteractionSpace instance
    into a readable dataframe.

    Parameters
    ----------
    ccc_matrix : pandas.DataFrame
        A dataframe containing the communication scores for a given combination
        between a pair of sender-receiver cells and a ligand-receptor pair.
        Columns are pairs of cells and rows LR pairs.

    Returns
    -------
    readable_ccc : pandas.DataFrame
        A dataframe containing flat information in each row about communication
        scores for a given pair of cells and a specific LR pair. A row contains
        the sender and receiver cells as well as the ligand and the receptor
        participating in an interaction and their respective communication
        score. Columns are: ['sender', 'receiver', 'ligand', 'receptor',
        'communication_score']
    '''
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
    '''Sorts cells by senders first and alphabetically and creates pairs of
    senders-ligands. If senders and receivers share cells, it creates pairs
    of senders-receptors for those shared cells. Then sorts receivers cells
    and creates pairs of receivers-receptors, for those cells that are not
    shared with senders.

    Parameters
    ----------
    sender_cells : list
        List of sender cells to sort.

    receiver_cells : list
        List of receiver cells to sort.

    ligands : list
        List of ligands to sort.

    receptors : list
        List of receptors to sort.

    Returns
    -------
    sorted_nodes : dict
        A dictionary where keys are the nodes of cells-proteins and values
        are the position they obtained (a ranking from 0 to N, where N is
        the total number of nodes).
    '''
    sorted_nodes = dict()
    count = 0

    both = set(sender_cells) & set(receiver_cells) # Intersection
    for c in sender_cells:
        for p in ligands:
            sorted_nodes[(c + '^' + p)] = count
            count += 1

        if c in both:
            for p in receptors:
                sorted_nodes[(c + '^' + p)] = count
                count += 1

    for c in receiver_cells:
        if c not in both:
            for p in receptors:
                sorted_nodes[(c + '^' + p)] = count
                count += 1
    return sorted_nodes


def determine_small_radius(coordinate_dict):
    '''Computes the radius of a circle whose diameter is the distance
    between the center of two nodes.

    Parameters
    ----------
    coordinate_dict : dict
        A dictionary containing the coordinates to plot each node.

    Returns
    -------
    radius : float
        The half of the distance between the center of two nodes.
    '''
    # Cartesian coordinates
    keys = list(coordinate_dict.keys())
    circle1 = np.asarray(coordinate_dict[keys[0]])
    circle2 = np.asarray(coordinate_dict[keys[1]])
    diff = circle1 - circle2
    distance = np.sqrt(diff[0]**2 + diff[1]**2)
    radius = distance / 2.0
    return radius


def _build_network(sender_cells, receiver_cells, ligands, receptors, sorted_nodes, readable_ccc, excluded_score=0):
    '''Generates a directed network given a subset of sender and receiver cells
    as well as a ligands and receptors.

    Parameters
    ----------
    sender_cells : list
        List of cells to be included as senders.

    receiver_cells : list
        List of cells to be included as receivers.

    ligands : list
        List of genes/proteins to be included as ligands produced by the
        sender cells.

    receptors : list
        List of genes/proteins to be included as receptors produced by the
        receiver cells.

    readable_ccc : pandas.DataFrame
        A dataframe containing ligand-receptor interactions by a pair
        of specific sender-receiver cells and their communication score
        as rows. Columns are ['sender', 'receiver', 'ligand', 'receptor',
        'communication_score']

    excluded_score : float, default=0
        Rows that have a communication score equal or lower to this will
        be dropped from the network.

    Returns
    -------
    G : networkx.DiGraph
        Directed graph that contains nodes as sender-cell^ligand
        and receiver-cell^receptor, and links that goes from a
        pair of sender-cell^ligand to a pair of receiver-cell^receptor.
    '''
    G = nx.DiGraph()
    df = readable_ccc.loc[(readable_ccc.sender.isin(sender_cells)) & \
                          (readable_ccc.receiver.isin(receiver_cells)) & \
                          (readable_ccc.ligand.isin(ligands)) & \
                          (readable_ccc.receptor.isin(receptors))
                          ]

    df = df.loc[df['communication_score'] > excluded_score]

    assert len(df) > 0, 'There are no links for the subset of cells and ligand-receptor pairs'

    for idx, row in df.iterrows():
        # Generate node names
        node1 = row.sender + '^' + row.ligand
        node2 = row.receiver + '^' + row.receptor

        # Add nodes to the network
        G.add_node(node1,
                   cell=row.sender,
                   protein=row.ligand,
                   signal='ligand',
                   sorting=int(sorted_nodes[node1]))

        G.add_node(node2,
                   cell=row.receiver,
                   protein=row.receptor,
                   signal='receptor',
                   sorting=int(sorted_nodes[node2]))

        # Add edges to. the network
        G.add_edge(node1, node2, weight=row.communication_score)
    return G


def get_node_colors(G, coloring_feature=None, cmap='viridis'):
    '''Generates colors for each node in a network given one of their
    properties.

    Parameters
    ----------
    G : networkx.Graph
        A graph containing a list of nodes

    coloring_feature : str, default=None
        A node attribute present in the dictionary associated with
        each node. The values associated with this attributed will
        be used for coloring the nodes.

    cmap : str, default='viridis'
        Name of a matplotlib color palette for coloring the nodes.

    Returns
    -------
    node_colors : dict
        A dictionary wherein each key is a node and values are
        tuples containing colors in the RGBA format.

    feature_colores : dict
        A dictionary wherein each key is a value for the attribute
        of nodes in the coloring_feature property and values are
        tuples containing colors in the RGBA format.
    '''
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
    '''Adds legends to circos plot.

    Parameters
    ----------
    cell_legend : dict
        Dictionary containing the colors for the cells.

    signal_legend : dict, default=None
        Dictionary containing the colors for the LR pairs in a given pair
        of cells. Corresponds to the colors of the links.

    meta_legend : dict, default=None
        Dictionary containing the colors for the cells given their major
        groups.

    fontsize : int, default=14
        Size of the labels in the legend.
    '''
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
