#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import math as m
import matplotlib.pyplot  as plt
import pandas as pd


def nearest_neighbor(coords, u_1=0):
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

    print(p_nodes)
    print("W: ", W)
    return W, p_nodes


def nn_best(coords):
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


def best_insertion(coords):
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

    print(p_nodes)
    print("W: ", W)
    return W, p_nodes


def bi_best(coords, rep):
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


if __name__ == '__main__':

    # prepare list of nodes
    graph_path = r'berlin.csv'
    graph = pd.read_csv(graph_path, delimiter=';')
    graph.apply(pd.to_numeric, errors='coerce').fillna(graph)
    graph_XY = graph[['POINT_X', 'POINT_Y']].values.tolist()
    graph_names = graph['nazev']

    # compute best weights and nodes for both methods
    weight_NN, nodes_NN = nn_best(graph_XY)
    weight_BI, nodes_BI = bi_best(graph_XY, len(graph_XY))

    # prepare lists for plotting
    x_nodes_NN, y_nodes_NN = list(zip(*nodes_NN))
    labels_NN = map(str, list(range(1, len(nodes_NN))))
    x_nodes_BI, y_nodes_BI = list(zip(*nodes_BI))
    labels_BI = map(str, list(range(1, len(nodes_BI))))

    # plot results of NN
    plt.figure()
    plt.scatter(x_nodes_NN, y_nodes_NN)
    plt.plot(x_nodes_NN, y_nodes_NN)
    for xy, i in zip(graph_XY, graph_names):
        plt.text(xy[0], xy[1], str(i), va='top', color='k', backgroundcolor='r')
    for x, y, i in zip(list(x_nodes_NN), list(y_nodes_NN), labels_NN):
        plt.text(x, y, i, va='bottom', color='r', backgroundcolor='k')
    plt.title('Nearest Neighbor, W=%s' % weight_NN)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # plot results of BI
    plt.figure()
    plt.scatter(x_nodes_BI, y_nodes_BI)
    plt.plot(x_nodes_BI, y_nodes_BI)
    for x, y, i in zip(list(x_nodes_BI), list(y_nodes_BI), labels_BI):
        plt.text(x, y, i)
    plt.title('Best insertion, W=%s' % weight_BI)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
