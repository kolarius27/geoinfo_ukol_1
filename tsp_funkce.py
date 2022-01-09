#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random


def tour_solution(coords, tour):
    """
    Function tour_solution generates Hamilton's cycle based on list of nodes and sequence.
    :param coords: list of XY coordinates
    :param tour: list of integers of indexes
    :return: sum of weights W (float), nodes in Hamilton's cycle (list)
    """

    # rearrange list of coordinates based on best tour
    xy_coords = [coords[i-1] for i in tour]
    xy_arr = [np.array(xy) for xy in xy_coords]

    # initialize sum of weights W
    W = 0

    # start position
    u = xy_arr.pop(0)

    # loop through all nodes
    while len(xy_arr) != 0:
        # second node of edge
        u_i = xy_arr.pop(0)

        # calculate weight of edge and add to sum of weights
        w_d = np.linalg.norm(u-u_i)
        W += w_d

        # move to another node
        u = u_i

    return W, xy_coords


def nearest_neighbor(coords, u_1=0):
    """
    Function nearest_neighbor generates Hamilton's cycle based on list of nodes and index of starting node
    using Nearest Neighbor heuristics.
    :param coords: list of XY coordinates
    :param u_1: integer, index of starting node, default is 0
    :return: sum of weights W (float), nodes in Hamilton's cycle (list)
    """
    # initialize unprocessed nodes, processed nodes and sum of weights W
    u_nodes = coords.copy()
    p_nodes = []
    W = 0

    # decide on first node
    if 0 <= u_1 <= len(u_nodes)-1:
        u_i = np.array(u_nodes.pop(u_1))
    else:
        random.shuffle(u_nodes)
        u_i = np.array(u_nodes.pop())

    # save starting position
    start = u_i

    # mark first node as processed node
    p_nodes.append(u_i)

    # go through unprocessed nodes until every node is processed
    while len(u_nodes) != 0:
        # initialize list of distances between u_i and every unprocessed node
        dists = []

        # calculate distances between u_i and every unprocessed node
        for node in u_nodes:
            u = np.array(node)
            dist = np.linalg.norm(u_i - u)
            dists.append(dist)

        # check for minimal distance and index of closest node to u_i
        min_dist = min(dists)
        min_i = dists.index(min_dist)

        # add distance to sum of weights W
        W += min_dist

        # set new u_i and mark it as processed node
        u_i = np.array(u_nodes.pop(min_i))
        p_nodes.append(u_i)

    # add distance between first and last node
    W += np.linalg.norm(start - u_i)

    # append starting position to end of list (plotting)
    p_nodes.append(start)

    return W, p_nodes


def best_insertion(coords):
    """
    Function best_insertion generates Hamilton's cycle based on list of nodes
    using Best Insertion heuristics.
    :param coords: list of XY coordinates
    :return: sum of weights W (float), nodes in Hamilton's cycle (list)
    """
    # initialize unprocessed nodes, processed nodes and sum of weights W
    u_nodes = coords.copy()
    p_nodes = []
    W = 0

    # pick three random points to initialize hamilton's circle
    for _ in range(3):
        random.shuffle(u_nodes)
        u_i = np.array(u_nodes.pop())
        p_nodes.append(u_i)

    # save starting position
    start = p_nodes[0]

    # calculate distances between first three nodes
    for i in range(len(p_nodes)):
        u1 = p_nodes[i]
        u2 = (p_nodes + [start])[i+1]
        dist = np.linalg.norm(u1 - u2)
        W += dist

    # go through unprocessed nodes until every node is processed
    while len(u_nodes) != 0:
        # catch out of index exception
        hamilton = p_nodes + [start]

        # pick random node
        random.shuffle(u_nodes)
        u = np.array(u_nodes.pop())

        # initialize list of delta w distances
        d_ws = []

        # calculating delta w for every edge
        for j in range(len(p_nodes)):
            d_w = np.linalg.norm(hamilton[j] - u) + np.linalg.norm(u - hamilton[j+1]) \
                   - np.linalg.norm(hamilton[j] - hamilton[j+1])
            d_ws.append(d_w)

        # check for minimal delta w and index of corresponding node
        min_w = min(d_ws)
        min_i = d_ws.index(min_w)

        # insert node to hamilton's circle
        p_nodes.insert(min_i+1, u)
        W += min_w

    # append starting position to end of list (plotting)
    p_nodes.append(start)

    return W, p_nodes


def nn_best(coords):
    """
    Function nn_best picks the best solution generated using Nearest Neighbor heuristics.
    :param coords: list of XY coordinates
    :return: sum of weights W (float), nodes in Hamilton's cycle (list)
    """
    # initialize list of sum of weights W and list of list of processed nodes
    W_list = []
    p_nodes_list = []

    # for every node as a start node compute Hamilton's cycle
    for k in range(len(coords)):
        W_k, p_nodes_k = nearest_neighbor(coords, k)
        W_list.append(W_k)
        p_nodes_list.append(p_nodes_k)

    # get minimal sum of weights and its index
    W_min = min(W_list)
    W_min_i = W_list.index(W_min)

    # search for Hamilton's cycle with minimal sum of weights
    p_nodes_min = p_nodes_list[W_min_i]

    return W_min, p_nodes_min


def bi_best(coords, rep):
    """
    Function bb_best picks the best solution generated using Nearest Neighbor heuristics.
    :param coords: list of XY coordinates
    :param rep: integer, number of repetitions
    :return: sum of weights W (float), nodes in Hamilton's cycle (list)
    """
    # initialize list of sum of weights W and list of list of processed nodes
    W_list = []
    p_nodes_list = []

    # for every node as a start node compute Hamilton's cycle
    for k in range(rep):
        W_k, p_nodes_k = best_insertion(coords)
        W_list.append(W_k)
        p_nodes_list.append(p_nodes_k)

    # get minimal sum of weights and its index
    W_min = min(W_list)
    W_min_i = W_list.index(W_min)

    # search for Hamilton's cycle with minimal sum of weights
    p_nodes_min = p_nodes_list[W_min_i]

    return W_min, p_nodes_min


def bi_tuning(coords, w_best, k_wanted):
    """
    Function bi_tuning computes Hamilton's based on list of nodes using Best Insertion heuristics repeatedly, until
    desired k-coefficient is fulfilled.
    :param coords: list of XY coordinates
    :param w_best: float, optimal length of Hamilton's cycle
    :param k_wanted: float, desired k-coefficient
    :return: number of repetitions (integer), evolution of minimizing k-coefficient (list), Hamilton's cycle (list)
    """
    # initialize list of k-coefficients, minimal k-coefficients, number of repetitions and repetition variable
    k_list = []
    k_min = []
    reps = 0

    # while condition is true
    while True:
        # new repetition
        reps += 1

        # compute sum of weights and Hamilton's cycle with Best Insertion method
        w_d, p_nodes = best_insertion(coords)

        # compute k-coefficient and check for minimal value
        k = w_d / w_best
        k_list.append(k)
        k_min.append(min(k_list))

        print(reps, "%.3f" % min(k_list))

        # if k is lower or equal to wanted k, while cycle stops
        if k <= k_wanted:

            return reps, k_min, p_nodes
