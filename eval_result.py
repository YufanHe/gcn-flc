import os
import numpy as np
import json
from gcn_flc.datautils import savedata, loaddata, graph_dis
from gcn_flc.graphutils import vis_graph, save_graph, load_adj
from matplotlib import pyplot as plt
import argparse
import datetime
import pdb
import multiprocessing
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='data generator')
parser.add_argument('--eval_file', dest='eval_file', default='./dataset/output/dataset-04-28-09-26-04_output.json', help='Result file from GCN to be evaluated')
parser.add_argument('--result_dir', dest='result_dir', default='./dataset/result/', help='Directory to save the evaluation result')
parser.add_argument('--parallel_core', dest='parallel_core', default=3, type=int, help='Parallel process to evaluate the result')

def main():
    global args
    args = parser.parse_args()
    cfg, data = loaddata(args.eval_file)
    # number of samples
    N = cfg['sample_num']
    # N = 10
    # Add shared data between processes
    data_save=multiprocessing.Manager().list()   
    lock= multiprocessing.Manager().Lock() 
    # Add thread pool 
    p = Pool(args.parallel_core)
    for i in range(N):
        p.apply_async(eval_multi, args=(data[i],data_save,lock,i)) 
    # Wait for all threads finish
    p.close()
    p.join()
    # File to be saved
    file_name = os.path.basename(args.eval_file)
    file_name = os.path.splitext(file_name)[0] + '_eval.json'
    file_save = os.path.join(args.result_dir,file_name)
    # Remove old file
    if os.path.exists(file_save):
        os.remove(file_save)
    # Open file to save
    with open(file_save, 'w') as fp:
        fp.write(json.dumps(list(data_save), indent=3))
    data_np = np.array(data_save)
    a = data_np[:,2]
    a = a[a < 10]
    print(a.mean())

def eval_multi(data_node,data_save,lock,idx):
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
    pd_mask = np.array(data_node['predict_x'])
    overall_mask = np.logical_or(gd_mask, pd_mask)

    facility_num = len(data_node['facilities'])
    client_num = len(data_node['clients'])
    node_num = facility_num + client_num

    # initialize the distance matrix
    dist_overall = 999999 * np.ones((client_num, facility_num))
    # load graph from the data
    G = load_adj(data_node['graph_dict'], node_num)
    for i in range(facility_num):
        # Only calculate the dist in possible facility
        if overall_mask[i]:
            for j in range(client_num):
                dist_overall[j, i] = graph_dis(G, j, i, facility_num)

    gd_dis_ma = np.ma.masked_array(dist_overall, \
                                   mask=np.broadcast_to(1 - gd_mask, (client_num, facility_num)))

    pd_dis_ma = np.ma.masked_array(dist_overall, \
                                   mask=np.broadcast_to(1 - pd_mask, (client_num, facility_num)))

    gd_dis = gd_dis_ma.min(axis=1)
    pd_dis = pd_dis_ma.min(axis=1)

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
          % (idx, gd_cost, pd_cost, pd_cost/gd_cost))

    with lock:
        data_save.append([gd_cost, pd_cost, pd_cost/gd_cost])


if __name__ == '__main__':
    main()
