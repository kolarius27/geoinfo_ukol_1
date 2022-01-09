#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import time
from tsp_funkce import *


if __name__ == '__main__':

    # prepare list of nodes
    graph_path = r'data\lin105.csv'
    graph = pd.read_csv(graph_path, delimiter=';')
    graph.apply(pd.to_numeric, errors='coerce').fillna(graph)
    graph_XY = graph[['POINT_X', 'POINT_Y']].values.tolist()

    # prepare sequence of optimal solution
    graph_tour_path = r'data\lin105_tour.csv'
    graph_tour = pd.read_csv(graph_tour_path, delimiter=';', header=None)
    graph_tour.apply(pd.to_numeric, errors='coerce')
    graph_tour = graph_tour[0].tolist()

    # prepare sequence of gis solutions
    graph_tour_gis_path1 = r'data\lin105_tour_gis_dt.csv'
    graph_tour_gis1 = pd.read_csv(graph_tour_gis_path1, delimiter=';', header=None)
    graph_tour_gis1.apply(pd.to_numeric, errors='coerce')
    graph_tour_gis1 = graph_tour_gis1[0].tolist()

    graph_tour_gis_path2 = r'data\lin105_tour_gis_all.csv'
    graph_tour_gis2 = pd.read_csv(graph_tour_gis_path2, delimiter=';', header=None)
    graph_tour_gis2.apply(pd.to_numeric, errors='coerce')
    graph_tour_gis2 = graph_tour_gis2[0].tolist()

    # compute best solution
    weight_best, nodes_best = tour_solution(graph_XY, graph_tour)

    # compute Hamilton's cycle from ArcGIS Pro Network Analyst
    weight_gis1, nodes_gis1 = tour_solution(graph_XY, graph_tour_gis1)
    weight_gis2, nodes_gis2 = tour_solution(graph_XY, graph_tour_gis2)

    # compute best weights and nodes for both methods
    start_NN = time.time()
    weight_NN, nodes_NN = nn_best(graph_XY)
    end_NN = time.time()
    time_NN = end_NN - start_NN

    start_BI = time.time()
    weight_BI, nodes_BI = bi_best(graph_XY, len(graph_XY))
    end_BI = time.time()
    time_BI = end_BI - start_BI

    # prepare lists for plotting
    x_nodes_best, y_nodes_best = list(zip(*nodes_best))
    x_nodes_NN, y_nodes_NN = list(zip(*nodes_NN))
    x_nodes_BI, y_nodes_BI = list(zip(*nodes_BI))
    x_nodes_gis1, y_nodes_gis1 = list(zip(*nodes_gis1))
    x_nodes_gis2, y_nodes_gis2 = list(zip(*nodes_gis2))

    # calculate k-coefficient
    k_NN = weight_NN / weight_best
    k_BI = weight_BI / weight_best
    k_gis1 = weight_gis1 / weight_best
    k_gis2 = weight_gis2 / weight_best
    k_gis = min(k_gis1, k_gis2)
    if k_gis == k_gis1:
        x_nodes_gis, y_nodes_gis = x_nodes_gis1, y_nodes_gis1
        weight_gis = weight_gis1
    else:
        x_nodes_gis, y_nodes_gis = x_nodes_gis2, y_nodes_gis2
        weight_gis = weight_gis2

    ### plotting results

    ## comparison of ArcGIS Pro results
    fig1, axs1 = plt.subplots(1, 2)
    fig1.suptitle('lin105 – ArcGIS Pro solutions')

    # Delaunay triangulation – axs[0, 0]
    axs1[0].scatter(x_nodes_gis1, y_nodes_gis1, color='black')
    axs1[0].plot(x_nodes_gis1, y_nodes_gis1, color='brown')
    axs1[0].scatter(x_nodes_gis1[0], y_nodes_gis1[0], s=40, color='brown')
    axs1[0].set_title('Edges: Delaunay triangulation, W=%.3f, k=%.3f, t=%.3f' % (weight_gis1, k_gis1, 5.))

    axs1[1].scatter(x_nodes_gis2, y_nodes_gis2, color='black')
    axs1[1].plot(x_nodes_gis2, y_nodes_gis2, color='orange')
    axs1[1].scatter(x_nodes_gis2[0], y_nodes_gis2[0], s=40, color='orange')
    axs1[1].set_title('Edges: All possible edges, W=%.3f, k=%.3f, t=%.3f' % (weight_gis2, k_gis2, 11.))

    ## comparison of all results
    fig2, axs2 = plt.subplots(2, 2)
    fig2.suptitle('lin105 – Comparison of all results')
    for a in axs2:
        for b in a:
            b.scatter(x_nodes_best, y_nodes_best, color='black')

    # Best solution – axs[0, 0]
    axs2[0, 0].plot(x_nodes_best, y_nodes_best)
    axs2[0, 0].scatter(x_nodes_best[0], y_nodes_best[0], s=40)
    axs2[0, 0].set_title('Optimal solution, W=%.3f, k=%.3f' % (weight_best, 1.000))

    # Nearest Neighbor – axs[0, 1]
    axs2[0, 1].plot(x_nodes_NN, y_nodes_NN, color='green')
    axs2[0, 1].scatter(x_nodes_NN[0], y_nodes_NN[0], color='green', s=40)
    axs2[0, 1].set_title('Nearest Neighbor, W=%.3f, k=%.3f, t=%.3f s' % (weight_NN, k_NN, time_NN))

    # Best Insertion – axs[1, 0]
    axs2[1, 0].plot(x_nodes_BI, y_nodes_BI, color='red')
    axs2[1, 0].scatter(x_nodes_BI[0], y_nodes_BI[0], color='red', s=40)
    axs2[1, 0].set_title('Best insertion, W=%.3f, k=%.3f, t=%.3f s' % (weight_BI, k_BI, time_BI))

    # Best solution – axs[0, 0]
    axs2[1, 1].plot(x_nodes_gis, y_nodes_gis, color='magenta')
    axs2[1, 1].scatter(x_nodes_gis[0], y_nodes_gis[0], s=40, color='magenta')
    axs2[1, 1].set_title('ArcGIS Pro Network Analyst, W=%.3f, k=%.3f, t=%.3f s' % (weight_gis, k_gis, 11.0))

    plt.show()

    # tuning of best insertion result
    plt.figure(3)
    t_start = time.time()
    bi_reps, bi_k, _ = bi_tuning(graph_XY, weight_best, 1.01)
    t_end = time.time()
    t_time = t_end - t_start
    print(bi_reps)
    x_reps = [i for i in range(1, bi_reps+1)]
    plt.plot(x_reps, bi_k)
    plt.title('Number of repetitions: %i, time: %.3f' % (bi_reps, t_time))
    plt.show()
