import os
import numpy as np
import json
from gcn_flc.datautils import savedata, loaddata, graph_dis, visualize
from gcn_flc.graphutils import vis_graph, save_graph, load_adj
from matplotlib import pyplot as plt
import argparse
import datetime
import pdb
import multiprocessing
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='data generator')
parser.add_argument('--eval_file', dest='eval_file', default='/data/pli32/trans/predict_5_1.json', help='Result file from GCN to be evaluated')
parser.add_argument('--result_dir', dest='result_dir', default='./dataset/result/', help='Directory to save the evaluation result')
parser.add_argument('--parallel_core', dest='parallel_core', default=16, type=int, help='Parallel process to evaluate the result')
parser.add_argument('--rand', dest='rand', default=False, type=int, help='Random node selection')

def main():
    global args
    args = parser.parse_args()
    cfg, data = loaddata(args.eval_file)
    Ni = 122
    eval_vis(data[Ni])
    

def eval_vis(data_node):
    """
    Multi_process function for result evaluation
    
    Args: 
        data_node: a node that contains all the information about the graph and 
                prediction and ground truth
        data_save: shared data list across the multi-process
        lock: a lock that make sure the operation is not conflict
    Return:
        
    """
    gd_mask = np.array(data_node['x'])
    if args.rand:
        pd_mask = np.round(np.random.random_sample(gd_mask.shape)*5.0/8.5)
        pd_mask = pd_mask.astype(int)
    else:
        pd_mask = np.array(data_node['predict_x'])
        overall_mask = np.logical_or(gd_mask, pd_mask)

    facility_num = len(data_node['facilities'])
    client_num = len(data_node['clients'])
    node_num = facility_num + client_num

    dist_overall = data_node['d']
    

    gd_dis_ma = np.ma.masked_array(dist_overall, \
                                   mask=np.broadcast_to(1 - gd_mask, (client_num, facility_num)))

    pd_dis_ma = np.ma.masked_array(dist_overall, \
                                   mask=np.broadcast_to(1 - pd_mask, (client_num, facility_num)))

    gd_dis = gd_dis_ma.min(axis=1)
    gd_assign = gd_dis_ma.argmin(axis=1)
    pd_dis = pd_dis_ma.min(axis=1)
    pd_assign = pd_dis_ma.argmin(axis=1)

    cost_overall = data_node['charge']

    c1 = np.ma.masked_array(cost_overall, mask=1 - gd_mask)
    gd_cost = gd_dis.data.sum() + c1.sum()
    c2 = np.ma.masked_array(cost_overall, mask=1 - pd_mask)
    if c2.all() is np.ma.masked:
        print("%03d: Not feasible" %(idx))
        pd_cost = 1e5
    else:
        pd_cost = pd_dis.data.sum() + c2.sum()
        print("%03d: The optimal cost is %0.2f, ther predicted cost is %0.2f , ratio is %0.2f" \
          % (999, gd_cost, pd_cost, pd_cost/gd_cost))
    fig = visualize(data_node, gd_mask, gd_assign, False)
    fig.show()
    input('any key to continue')
    fig.clear()
    fig = visualize(data_node, gd_mask, pd_assign, False)
    fig.show()
    input('any key to continue')
    fig.clear()



if __name__ == '__main__':
    main()
