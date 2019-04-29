import networkx as nx
import pylab 
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.sparse import coo_matrix

def euc_dis(a,b):
    '''
    Euclidean distance between any two point
    Args: 
        a,b numpy array with same dim (both 1*n vector)
    Return:
        distance between a and b
    '''
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)

def graph_generation(facilities,clients,max_dis = 18,reduce_prob = 0.4):
    '''
    Given position of the facilities and clients,
    generate a graph that connects all the nodes but
    not fully connected
    Args:
        facilities: position array of facilities n*2
        clinets:   position array of clients
    Return:
        G: fully connected graph
        T: minimum spanning tree of the graph
        TG: Minimum spanning tree with some random edges
    '''

    fac_num = facilities.shape[0]
    cli_num = clients.shape[0]

    #combine nodes
    nodes = np.concatenate((facilities,clients),axis = 0)

    #Initialize the graph
    G=nx.Graph()
    N = fac_num + cli_num
    # add nodes
    G.add_nodes_from(np.arange(0,N).tolist())

    # suppose bidirectional edges
    for i in range(N):
        for j in range(i+1,N):
            dis = euc_dis(nodes[i],nodes[j])
            G.add_edge(i, j, weight=dis)
    
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
    
    return G,T,TG

def save_graph(G,fac_num,cli_num):
    '''
    Save a graph to an adjacent matrix
    store the sparse matrix as coo format and 
    put it in a dict
    Args:
        G: graph in networkx
        fac_num: number of facilities
        cli_num: number of clients
    Return:
        g_dict: a dict that stores the sparse matrix
    '''

    # Convert to adjacent matrix
    A = nx.adjacency_matrix(G)
    Ac = A.tocoo()
    g_dict = {}
    g_dict['data'] = Ac.data.tolist()
    g_dict['row'] = Ac.row.tolist()
    g_dict['col'] = Ac.col.tolist()

    return g_dict

def load_adj(g_dict,N):
    '''
    load graph from adjacent matrix
    Args:
        g_dict: a dict that stores the sparse matrix
        N: the number of nodes
    Return:
        G: graph represented by Networkx
    '''
    row  = np.array(g_dict['row'])
    col  = np.array(g_dict['col'])
    data = np.array(g_dict['data'])
    A = coo_matrix((data, (row, col)), shape=(N, N)).toarray()
    G = nx.Graph(A)

    return G


def vis_graph(nodes,graph,fac_num,cli_num):
    '''
    visualize the graph
    Args:
        nodes: positions of each node in order to align layout with the real position
        graph: the graph you want to visulize
        fac_num: number of facilities to seperate them from others
        cli_num: number of clinets
    '''

    N = nodes.shape[0]
    #color_map
    color_map = []
    for node in graph:
        if node < fac_num:
            color_map.append('blue')
        else: color_map.append('green') 
    fixed_positions =  {i: tuple(nodes[i]) for i in range(N)}
    fixed_nodes = fixed_positions.keys()
    # Generate layout
    pos_G=nx.spring_layout(graph,pos=fixed_positions, fixed = fixed_nodes)
    nx.draw(graph,pos_G,with_labels=False, node_color=color_map, edge_color='red', node_size=30, alpha=0.5,width=0.5)


def test():
    '''
        Some test code for developing
        Not needed for the generator code
    '''
    fac_num = 8
    cli_num = 92
    N = fac_num + cli_num

    world_size = np.array([100,100])
    #generate facilities
    facilities = np.random.rand(fac_num,2)*world_size[None,:]
    #generate clients
    clients = np.random.rand(cli_num,2)*world_size[None,:]
    #combine nodes
    nodes = np.concatenate((facilities,clients),axis = 0)

    G,T,TG = graph_generation(facilities,clients)
    #G,_,_ = graph_generation(facilities,clients)
    
    dict_TG = save_graph(TG,fac_num,cli_num)
    TGr = load_adj(dict_TG, N)
    fig=plt.figure(figsize=(16, 8))

    fig.add_subplot(1, 2, 1)
    # nx.draw(TG,pos_TG,with_labels=False, node_color=color_map, edge_color='red', node_size=30, alpha=0.5,width=0.5)
    vis_graph(nodes,TG,fac_num,cli_num)
    fig.add_subplot(1, 2, 2)
    # nx.draw(TG,pos_TG,with_labels=False, node_color=color_map, edge_color='red', node_size=30, alpha=0.5,width=0.5)
    vis_graph(nodes,TGr,fac_num,cli_num)
    fig.show()
    input('any key to continue')
    

    # color map for graph
    color_map = []
    for node in G:
        if node < fac_num:
            color_map.append('blue')
        else: color_map.append('green') 

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
    # nx.draw(TG,pos_TG,with_labels=False, node_color=color_map, edge_color='red', node_size=30, alpha=0.5,width=0.5)
    vis_graph(nodes,TG,fac_num,cli_num)
    plt.title('Add more edges',fontsize=15)

    fig.add_subplot(2, 2, 4)
    plt.scatter(clients[:,0], clients[:,1], c = 'b')
    plt.scatter(facilities[:,0], facilities[:,1], c = 'r')
    plt.title('Scatters',fontsize=15)
    fig.show()
    input('any key to continue')
    # '''
    # Shortest Path with dijkstra_path
    # '''
    print('shortest part with dijkstra algorithm')
    path=nx.dijkstra_path(T, source=0, target=10)
    print('path from node 0 to node 10 in T', path)
    path=nx.dijkstra_path(TG, source=0, target=10)
    print('path from node 0 to node 10 in TG', path)

    #print('shortest path length with dijkstra algorithm')
    # distance=nx.dijkstra_path_length(T, source=0, target=10)
    # print('distance from 0 to 10', distance)


if __name__ == '__main__':
    test()



