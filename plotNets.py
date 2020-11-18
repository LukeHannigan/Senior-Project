import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('seaborn-dark-palette')
rcParams['font.family'] = 'serif'
from matplotlib.patches import Rectangle
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
matrix = 5
numBonds = int( (N**2 - N) / 2 )

# optimalNet_4g3.csv | 4 nodes with global coupling, net coupling of 3
data_name = 'optimalNet_4g3_cross%d.csv' % matrix
frame = pd.read_csv(os.path.join(base_dir, data_name))
frame = frame.query('Accept == True').iloc[-1]
data_arr = frame.to_numpy()

prob = frame[1]
err = frame[2]
var = (prob*100, err*100)

j_list = []
ind = 4 # 3 for run 1 only
for row in range(0, numBonds):
	for step in range(0, N - row - 1):
		col = row + step + 1
		J[row][col] = frame[ind] 
		J[col][row] = frame[ind]
		j_list.append(frame[ind])
		ind += 1

print(J)

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

txt = 'P(sync) = ({0:.2f} $\pm$ {1:.2f})%'
ax.annotate(txt.format(*var), xy=(0., .9), xycoords='axes fraction', fontsize=14)
fig.tight_layout()

''' optimalNet_4g3_graph1_temp.eps | run 1 of 4 node global network with net coupling of 3 '''
fig_name = 'optimalNet_4g3_graph%d_temp.eps' % matrix
fig.savefig(os.path.join(base_dir, fig_name))

plt.show()


