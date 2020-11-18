import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('seaborn-dark-palette')
rcParams['font.family'] = 'serif'
import networkx as nx
import numpy as np
import pandas as pd
import os

base_dir = '/home/luke/final_project/optimalNetwork'

J = np.array([[0., 0., 1.5, 0.],
      		  [0., 0., 0., 1.5],
		      [1.5, 0., 0., 0.],
			  [0., 1.5, 0., 0.]])
N = 4
margin = 5e-4
trials = 100
dur = 50
numBonds = int( (N**2 - N) / 2 )

j_list = []
for row in range(0, numBonds):
	for step in range(0, N - row - 1):
		col = row + step + 1
		j_list.append(J[col][row])

print(J)
print(j_list)

graph = nx.Graph()
i = 0
for row in range(0, numBonds):
	for step in range(0, N - row - 1):
		col = row + step + 1
		graph.add_edge(row+1, col+1, weight=j_list[i])
		i += 1

print(graph.nodes)

for edge in graph.edges:
	print(graph.get_edge_data(*edge))

pos = nx.circular_layout(graph) 
#cols = np.log10(j_list)
cols = j_list
cmap = plt.cm.viridis
jmin = 0. #min(cols)
jmax = 1.5 #max(cols) 

fig, ax = plt.subplots(figsize=(8, 8))

nx.draw(graph, pos, node_color='#787878', edge_color=cols, width=4, edge_cmap=cmap, with_labels=False, edge_vmin=jmin, edge_vmax=jmax)
nx.draw_networkx_labels(graph, pos, font_color='white')

scalarMap = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=jmin, vmax=jmax))
scalarMap._A = []
cbar = fig.colorbar(scalarMap, orientation='horizontal', pad=0.05)
cbar.ax.set_title('Bond Strength')

txt = 'P(sync) = 0%'
ax.annotate(txt, xy=(0.1, .9), xycoords='axes fraction', fontsize=14)
fig.tight_layout()
plt.show()

''' optimalNet_4g3_graph1_temp.eps | run 1 of 4 node global network with net coupling of 3 '''
fig_name = 'optimalNet_4g3_graph0_temp.eps'
fig.savefig(os.path.join(base_dir, fig_name))


