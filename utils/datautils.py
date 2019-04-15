from gurobipy import *
import math
from matplotlib import pyplot as plt
import numpy as np
import pdb

def distance(a,b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx*dx + dy*dy)
    
def gurobi_solver(data):
    """
    Exact solver for facility location
    Args:
    data.keys(): 
    clients: [N,2] list, e.g. [[0, 1.5],[2.5, 1.2]]
    facilities: [M,2] list, e.g. [[0,0],[0,1],[0,1],
                                 [1,0],[1,1],[1,2],
                                 [2,0],[2,1],[2,2]]
    charge: [M,2] list, e.g. [3,2,3,1,3,3,4,3,2]
    alpha: const, cost per mile
    Return:
    x: [M] binary array
    y: [N] scalar array
    d: [N,M] distance array
    """
    # Problem data
    clients = data['clients']
    facilities = data['facilities']
    charge = data['charge']
    alpha = data['alpha']

    numFacilities = len(facilities)
    numClients = len(clients)

    m = Model()

    # Add variables
    x = {}
    y = {}
    d = {} # Distance matrix (not a variable)
    

    for j in range(numFacilities):
        x[j] = m.addVar(vtype=GRB.BINARY, name="x%d" % j)

    for i in range(numClients):
        for j in range(numFacilities):
            y[(i,j)] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t%d,%d" % (i,j))
            d[(i,j)] = distance(clients[i], facilities[j])

    m.update()

    # Add constraints
    for i in range(numClients):
        for j in range(numFacilities):
            m.addConstr(y[(i,j)] <= x[j])

    for i in range(numClients):
        m.addConstr(quicksum(y[(i,j)] for j in range(numFacilities)) == 1)

    m.setObjective( quicksum(charge[j]*x[j] + quicksum(alpha*d[(i,j)]*y[(i,j)]
                    for i in range(numClients)) for j in range(numFacilities)) )

    m.optimize()
    # return a stardard result list
    x_result = []
    for j in range(numFacilities):
        x_result.append(x[j].X)
    y_result = []
    for i in range(numClients):
        for j in range(numFacilities):
            if y[(i,j)].X == 1:
                y_result.append(j)
                continue
    d_results = np.zeros((numClients, numFacilities))
    for i in range(numClients):
        for j in range(numFacilities):
            d_results[i,j] = d[(i,j)]
    return x_result, y_result, d_results

def visualize(data, x, y, vis=True):
    clients = np.array(data['clients'])
    facilities = np.array(data['facilities'])
    charge = np.array(data['charge'])
    alpha = data['alpha']

    plt.scatter(clients[:,0],clients[:,1])
    color = np.vstack([1/(max(charge)/charge)] + \
                      [[0]*len(charge)] + [[0]*len(charge)]).T
    plt.scatter(facilities[:,0],facilities[:,1],c=color)
    for i in range(len(y)):
        plt.plot([clients[i,0],facilities[y[i],0]],
                 [clients[i,1],facilities[y[i],1]],
                 c=[y[i]/len(facilities),
                    y[i]/len(facilities),
                    y[i]/len(facilities)])
    if vis:
        plt.show()
    return plt.gcf()
    

if __name__ == '__main__':
    data = {
        'clients': [[0, 1.5],[2.5, 1.2]],
        'facilities': [[0,0],[0,1],[0,1],
                     [1,0],[1,1],[1,2],
                     [2,0],[2,1],[2,2]],
        'charge': [3,2,3,1,3,3,4,3,2],
        'alpha': 10
    }
    x, y, d = gurobi_solver(data)
    visualize(data, x, y)