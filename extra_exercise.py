#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import time
from tsp_funkce_extra import *


# prepare list of nodes
graph_path = r'data\lin105.csv'
graph = pd.read_csv(graph_path, delimiter=';')
graph.apply(pd.to_numeric, errors='coerce').fillna(graph)
graph_XY = graph[['POINT_X', 'POINT_Y']].values.tolist()

# calculate result of non-deterministic Best Insertion
start_BI = time.time()
weight_BI, nodes_BI, init_BI = bi_best(graph_XY, 'random', False, 30)
end_BI = time.time()
time_BI = end_BI - start_BI

# calculate result of non-deterministic Best Insertion with initialized Hamiltonian cycle by bounding box
start_BI_bb = time.time()
weight_BI_bb, nodes_BI_bb, init_BI_bb = bi_best(graph_XY, 'bb_box', False, 30)
end_BI_bb = time.time()
time_BI_bb = end_BI_bb - start_BI_bb

# calculate result of non-deterministic Best Insertion with initialized Hamiltonian cycle by bounding box
start_BI_ch = time.time()
weight_BI_ch, nodes_BI_ch, init_BI_ch = bi_best(graph_XY, 'convex_hull', False, 30)
end_BI_ch = time.time()
time_BI_ch = end_BI_ch - start_BI_ch

# calculate result of deterministic Best Insertion
start_BI_d = time.time()
weight_BI_d, nodes_BI_d, init_BI_d = bi_best(graph_XY, 'random', True, 30)
end_BI_d = time.time()
time_BI_d = end_BI_d - start_BI_d

# calculate result of deterministic Best Insertion with initialized Hamiltonian cycle by bounding box
start_BI_d_bb = time.time()
weight_BI_d_bb, nodes_BI_d_bb, init_BI_d_bb = bi_best(graph_XY, 'bb_box', True, 1)
end_BI_d_bb = time.time()
time_BI_d_bb = end_BI_d_bb - start_BI_d_bb

# calculate result of deterministic Best Insertion with initialized Hamiltonian cycle by bounding box
start_BI_d_ch = time.time()
weight_BI_d_ch, nodes_BI_d_ch, init_BI_d_ch = bi_best(graph_XY, 'convex_hull', True, 1)
end_BI_d_ch = time.time()
time_BI_d_ch = end_BI_d_ch - start_BI_d_ch

# prepare Hamiltonian cycles for plotting
x_nodes_BI, y_nodes_BI = list(zip(*nodes_BI))
x_nodes_BI_bb, y_nodes_BI_bb = list(zip(*nodes_BI_bb))
x_nodes_BI_ch, y_nodes_BI_ch = list(zip(*nodes_BI_ch))
x_nodes_BI_d, y_nodes_BI_d = list(zip(*nodes_BI_d))
x_nodes_BI_d_bb, y_nodes_BI_d_bb = list(zip(*nodes_BI_d_bb))
x_nodes_BI_d_ch, y_nodes_BI_d_ch = list(zip(*nodes_BI_d_ch))

# prepare initial Hamiltonian cycles for plotting
x_init_BI, y_init_BI = list(zip(*init_BI))
x_init_BI_bb, y_init_BI_bb = list(zip(*init_BI_bb))
x_init_BI_ch, y_init_BI_ch = list(zip(*init_BI_ch))
x_init_BI_d, y_init_BI_d = list(zip(*init_BI_d))
x_init_BI_d_bb, y_init_BI_d_bb = list(zip(*init_BI_d_bb))
x_init_BI_d_ch, y_init_BI_d_ch = list(zip(*init_BI_d_ch))

# calculate k-coefficients
# weight_best = 6110.860949680391
weight_best = 14382.99593345118
k_BI = weight_BI / weight_best
k_BI_bb = weight_BI_bb / weight_best
k_BI_ch = weight_BI_ch / weight_best
k_BI_d = weight_BI_d / weight_best
k_BI_d_bb = weight_BI_d_bb / weight_best
k_BI_d_ch = weight_BI_d_ch / weight_best

# comparison of all results
fig1, axs1 = plt.subplots(2, 3)
fig1.suptitle('lin105 – Comparison of Best Insertion methods')
for a in axs1:
    for b in a:
        b.scatter(x_nodes_BI, y_nodes_BI, color='black')

# Best insertion – axs[0, 0]
axs1[0, 0].plot(x_nodes_BI, y_nodes_BI)
axs1[0, 0].scatter(x_init_BI, y_init_BI, s=40)
axs1[0, 0].set_title('Best Insertion, W=%.3f, k=%.3f, t=%.3f s' % (weight_BI, k_BI, time_BI))

# Best Insertion (bb-box) – axs[0, 1]
axs1[0, 1].plot(x_nodes_BI_bb, y_nodes_BI_bb, color='green')
axs1[0, 1].scatter(x_init_BI_bb, y_init_BI_bb, color='green', s=40)
axs1[0, 1].set_title('BI (bb-box), W=%.3f, k=%.3f, t=%.3f s' % (weight_BI_bb, k_BI_bb, time_BI_bb))

# Best Insertion (convex hull) – axs[0, 2]
axs1[0, 2].plot(x_nodes_BI_ch, y_nodes_BI_ch, color='orange')
axs1[0, 2].scatter(x_init_BI_ch, y_init_BI_ch, color='orange', s=40)
axs1[0, 2].set_title('BI (convex hull), W=%.3f, k=%.3f, t=%.3f s' % (weight_BI_ch, k_BI_ch, time_BI_ch))

# Deterministic Best Insertion – axs[1, 0]
axs1[1, 0].plot(x_nodes_BI_d, y_nodes_BI_d, color='red')
axs1[1, 0].scatter(x_init_BI_d, y_init_BI_d, color='red', s=40)
axs1[1, 0].set_title('Deter. BI, W=%.3f, k=%.3f, t=%.3f s' % (weight_BI_d, k_BI_d, time_BI_d))

# Deterministic Best Insertion (bb-box) – axs[1, 1]
axs1[1, 1].plot(x_nodes_BI_d_bb, y_nodes_BI_d_bb, color='magenta')
axs1[1, 1].scatter(x_init_BI_d_bb, y_init_BI_d_bb, s=40, color='magenta')
axs1[1, 1].set_title('Deter. BI (bb-box), W=%.3f, k=%.3f, t=%.3f s' % (weight_BI_d_bb, k_BI_d_bb, time_BI_d_bb))

# Deterministic Best Insertion (bb-box) – axs[1, 1]
axs1[1, 2].plot(x_nodes_BI_d_ch, y_nodes_BI_d_ch, color='brown')
axs1[1, 2].scatter(x_init_BI_d_ch, y_init_BI_d_ch, s=40, color='brown')
axs1[1, 2].set_title('Deter. BI (convex hull), W=%.3f, k=%.3f, t=%.3f s' % (weight_BI_d_ch, k_BI_d_ch, time_BI_d_ch))

plt.show()
