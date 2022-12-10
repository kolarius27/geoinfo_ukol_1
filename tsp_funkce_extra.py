#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import math as m
from scipy.spatial import ConvexHull


def bounding_box(u_nodes):
    x_coord, y_coord = zip(*u_nodes)
    indexes = [i for i, x in enumerate(x_coord)]
    x_max = [i for _, x, i in sorted(zip(y_coord, x_coord, indexes), reverse=True) if x == max(x_coord)]
    x_min = [i for _, x, i in sorted(zip(y_coord, x_coord, indexes)) if x == min(x_coord)]
    y_max = [i for _, y, i in sorted(zip(x_coord, y_coord, indexes)) if y == max(y_coord)]
    y_min = [i for _, y, i in sorted(zip(x_coord, y_coord, indexes), reverse=True) if y == min(y_coord)]
    minmax = x_min + y_max + x_max + y_min
    bb_box = []
    for m in minmax:
        if m not in bb_box:
            bb_box.append(m)
    p_nodes = [np.array(u_nodes[i]) for i in bb_box]
    for node in sorted(bb_box, reverse=True):
        u_nodes.pop(node)
    return p_nodes


def initialization(u_nodes, init_type):
    # initialize list of processed nodes and sum of weights W
    p_nodes = []
    W = 0

    # initialize Hamiltonian circle by three random points, bounding box or convex hull
    if init_type == 'random':
        for _ in range(3):
            random.shuffle(u_nodes)
            u_i = np.array(u_nodes.pop())
            p_nodes.append(u_i)
    elif init_type == 'bb_box':
        p_nodes = bounding_box(u_nodes)
    elif init_type == 'convex_hull':
        hull = ConvexHull(u_nodes)
        p_nodes = [np.array(u_nodes[vertex]) for vertex in hull.vertices]
        for vertex in sorted(hull.vertices, reverse=True):
            u_nodes.pop(vertex)

    # calculate distances between initialization nodes
    for i in range(len(p_nodes)):
        u1 = p_nodes[i]
        u2 = (p_nodes + [p_nodes[0]])[i+1]
        dist = np.linalg.norm(u1 - u2)
        W += dist

    return W, p_nodes


def bi_random(u_nodes, hamilton):
    # pick random node
    random.shuffle(u_nodes)
    u = np.array(u_nodes.pop())

    # initialize list of delta w distances
    d_ws = []

    # calculating delta w for every edge
    for j in range(len(hamilton)-1):
        d_w = np.linalg.norm(hamilton[j] - u) + np.linalg.norm(u - hamilton[j + 1]) \
              - np.linalg.norm(hamilton[j] - hamilton[j + 1])
        d_ws.append(d_w)

    # check for minimal delta w and index of corresponding node
    w_min = min(d_ws)
    i_min_u = d_ws.index(w_min) + 1

    return u, i_min_u, w_min


def bi_deter(u_nodes, hamilton):
    # initialize list of delta w distances and indexes for best delta w and best node
    w_min = m.inf
    i_min_u = 0
    i_node = 0

    # For every node
    for i in range(len(u_nodes)):
        u = np.array(u_nodes[i])
        d_ws = []
        # calculating delta w for every edge
        for j in range(len(hamilton)-1):
            d_w = np.linalg.norm(hamilton[j] - u) + np.linalg.norm(u - hamilton[j + 1]) \
                  - np.linalg.norm(hamilton[j] - hamilton[j + 1])
            d_ws.append(d_w)

        # if minimal delta w is smaller than before, update minimal distance and get new indexes
        if w_min > min(d_ws):
            w_min = min(d_ws)
            i_min_u = d_ws.index(w_min) + 1
            i_node = i

    # get unprocessed node that fits the best into Hamiltonian cycle
    u_min = np.array(u_nodes.pop(i_node))
    #random.shuffle(u_nodes)

    return u_min, i_min_u, w_min


def best_insertion(coords, init_type, deter=False):
    """
    Function best_insertion generates Hamiltonian cycle based on list of nodes
    using Best Insertion heuristics.
    :param coords: list of XY coordinates
    :return: sum of weights W (float), nodes in Hamiltonian cycle (list)
    """
    # initialize unprocessed nodes, processed nodes and sum of weights W
    u_nodes = coords.copy()

    # initialize Hamiltonian circle by three random points, bounding box or convex hull
    W, p_nodes = initialization(u_nodes, init_type)

    # save initial Hamilton cycle
    init_nodes = p_nodes.copy()

    # save starting position
    start = p_nodes[0]

    # go through unprocessed nodes until every node is processed
    while len(u_nodes) != 0:
        # catch out of index exception
        hamilton = p_nodes + [start]

        if deter is False:
            u, i_min_u, w_min = bi_random(u_nodes, hamilton)
        else:
            u, i_min_u, w_min = bi_deter(u_nodes, hamilton)

        # insert node to Hamiltonian circle
        p_nodes.insert(i_min_u, u)
        W += w_min

    # append starting position to end of list (plotting)
    p_nodes.append(start)

    return W, p_nodes, init_nodes


def bi_best(coords, init_type, deter, rep):
    """
    Function bb_best picks the best solution generated using Best Insertion heuristics.
    :param coords: list of XY coordinates
    :param rep: integer, number of repetitions
    :return: sum of weights W (float), nodes in Hamiltonian cycle (list)
    """
    # initialize list of sum of weights W and list of list of processed nodes
    W_list = []
    p_nodes_list = []
    init_nodes_list = []

    # for every node as a start node compute Hamiltonian cycle
    for k in range(rep):
        W_k, p_nodes_k, init_nodes_k = best_insertion(coords, init_type, deter)
        W_list.append(W_k)
        p_nodes_list.append(p_nodes_k)
        init_nodes_list.append(init_nodes_k)

    # get minimal sum of weights and its index
    W_min = min(W_list)
    W_min_i = W_list.index(W_min)

    # search for Hamiltonian cycle with minimal sum of weights
    p_nodes_min = p_nodes_list[W_min_i]
    init_nodes_min = init_nodes_list[W_min_i]
    print("Finished!")
    return W_min, p_nodes_min, init_nodes_min
