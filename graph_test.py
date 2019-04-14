import networkx as nx
import pylab 
from matplotlib import pyplot as plt
import numpy as np
import math

def euc_dis(a,b):
    '''
    Generate random samples
    Args: 
        a,b numpy array with same dim (both 1*n vector)
    Return:
        distance between a and b
    '''
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

fac_num = 8
cli_num = 92
max_dis = 5
reduce_prob = 0.4

world_size = np.array([20,20])
#generate facilities
facilities = np.random.rand(fac_num,2)*world_size[None,:]
#generate clients
clients = np.random.rand(cli_num,2)*world_size[None,:]
#combine nodes
nodes = np.concatenate((facilities,clients),axis = 0)

G=nx.Graph()
N = fac_num + cli_num
# add nodes
G.add_nodes_from(np.arange(0,N).tolist())

# suppose bidirectional edges
for i in range(N):
    for j in range(i+1,N):
        dis = euc_dis(nodes[i],nodes[j])
        G.add_edge(i, j, weight=dis)

A = nx.adjacency_matrix(G)
print(A)
# Minimum spanning tree
T=nx.minimum_spanning_tree(G)
# More connection tree
TG = T.copy()
# suppose bidirectional edges
for i in range(N):
    for j in range(i+1,N):
        dis = euc_dis(nodes[i],nodes[j])
        if dis < max_dis and np.random.random() > reduce_prob:
            TG.add_edge(i, j, weight=dis)

# color map for graph
color_map = []
for node in G:
    if node < fac_num:
        color_map.append('blue')
    else: color_map.append('green') 
for i in range(0,N):
    G.node[i]['pos'] = nodes[i]

# Apply fixed position layout for graph
fixed_positions =  {i: tuple(nodes[i]) for i in range(N)}
fixed_nodes = fixed_positions.keys()
# Generate layout
pos_G=nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
pos_T=nx.spring_layout(T,pos=fixed_positions, fixed = fixed_nodes)
pos_TG=nx.spring_layout(TG,pos=fixed_positions, fixed = fixed_nodes)

print('draw graph')
fig=plt.figure(figsize=(16, 16))
# fig.patch.set_facecolor('xkcd:mint green')

# draw graph
fig.add_subplot(2, 2, 1)
nx.draw(G,pos_G,with_labels=True, node_color=color_map, edge_color='red', node_size=200, alpha=0.5, fontsize = 8 )
plt.title('Full Graph',fontsize=15)

fig.add_subplot(2, 2, 2)
nx.draw(T,pos_T,with_labels=True, node_color=color_map, edge_color='red', node_size=100, alpha=0.5, fontsize = 4 )
plt.title('Minimum Spanning Tree',fontsize=15)


fig.add_subplot(2, 2, 3)
nx.draw(TG,pos_TG,with_labels=False, node_color=color_map, edge_color='red', node_size=30, alpha=0.5,width=0.5)
plt.title('Add more edges',fontsize=15)

fig.add_subplot(2, 2, 4)
plt.scatter(clients[:,0], clients[:,1], c = 'b')
plt.scatter(facilities[:,0], facilities[:,1], c = 'r')
plt.title('Scatters',fontsize=15)
plt.show()



# '''
# Shortest Path with dijkstra_path
# '''
# print('shortest part with dijkstra algorithm')
# path=nx.dijkstra_path(G, source=0, target=7)
# print('path from node 0 to node 7', path)
# print('shortest path length with dijkstra algorithm')
# distance=nx.dijkstra_path_length(G, source=0, target=7)
# print('distance from 0 to 7', distance)

