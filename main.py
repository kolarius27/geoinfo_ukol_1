#!/usr/bin/env python
# -*- coding: utf-8 -*-

from graph_def import *
import random
import numpy as np
import math as m


def nearest_neigbor(coords):
    u_nodes = coords
    p_nodes = []
    W = 0

    u_i = np.array(u_nodes.pop(random.randint(0, len(u_nodes))))
    p_nodes.append(u_i)
    print(u_nodes)
    print(u_i)
    print(p_nodes)
    while len(u_nodes) != 0:
        dist = m.inf
        for node in u_nodes:
            u = np.array(node)
            dist = np.linalg.norm(u_i-u)


def best_insertion(graph, coords):
    pass


if __name__ == '__main__':
    graph = G
    coords = C

    nearest_neigbor(graph, coords)
