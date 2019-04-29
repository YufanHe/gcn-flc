import os
import numpy as np
import json
from gcn_flc.datautils import gurobi_solver, visualize
from gcn_flc.graphutils import graph_generation, vis_graph, save_graph
from matplotlib import pyplot as plt
import argparse
import datetime
import pdb
import multiprocessing
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='data generator')
parser.add_argument('--config', dest='config', default='config.json',
                    help='hyperparameters')

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    return cfg

def data_groundtruth_process(cfg,data,lock):
    """
    Generate random samples
    Args: 
    cfg: configuration
    cfg.keys():
        total_nodes: total nodes of the graph, must equal GCN input dim
        facility_num: range of facility number
        world_size: Euclidean world grid size
        random_seed: random seed for generating a batch of training data
        sample_num: number of samples to generate
        facility_cost: range of facility opening cost
        travel_cost: the cost per euclidean distance from client to facility
    Return:
    data: list of dictionaries
    data[i].keys():
        clients: [N,2] list
        facilitis: [M,2] list
        charge: [M] list
        alpha: scalar
        x: [M] binary array, 1 if open this facility
        y: [N] scalar array, y[i] is the connected facility index
        d: [N,M] distance array, d[i,j] is the distance from i to j
    """

    np.random.seed()
    f_num = np.random.randint(cfg['facility_num'][0],
                                  cfg['facility_num'][1])
    facilities = list(map(list, 
                        list(zip(np.random.rand(f_num)*cfg['world_size'][0],
                                np.random.rand(f_num)*cfg['world_size'][1]))))   
    charge = np.random.randint(cfg['facility_cost'][0],
                                cfg['facility_cost'][1], f_num).tolist()
    c_num = cfg['total_nodes'] - f_num
    clients = list(map(list, 
                        list(zip(np.random.rand(c_num)*cfg['world_size'][0],
                                np.random.rand(c_num)*cfg['world_size'][1]))))
    alpha = cfg['travel_cost']   

    _,_,gen_graph = graph_generation(np.array(facilities),np.array(clients))
    graph_dict = save_graph(gen_graph, f_num, c_num)
    
    data_node = {
        'clients': clients,
        'facilities': facilities,
        'charge': charge,
        'alpha': alpha,
        'graph_dict':graph_dict
    }
    
    x, y, d = gurobi_solver(data_node)
    data_node['x'] = x
    data_node['y'] = y
    data_node['d'] = d.tolist()
    lock.acquire()
    data.append(data_node)
    print('Num of clients: %d' % len(clients))
    print('Generating the %d data'%(len(data)))
    lock.release()

def generate_data(cfg):

    #data = []
    data=multiprocessing.Manager().list()   
    lock= multiprocessing.Manager().Lock()  
    p = Pool(cfg['parallel_core'])
    #np.random.seed(cfg['random_seed'])
    for s in range(cfg['sample_num']):
        #data_groundtruth(cfg,data)  
        p.apply_async(data_groundtruth_process, args=(cfg,data,lock)) 
    p.close()
    p.join()
    return data

def savedata(data, cfg, name=None,data_dir = 'dataset/synthetic'):
    """
    Data will be saved in json format
    save the configuration and data both
    in the json file.
    
    Args: 
        data: data generated with optimal solutions
        cfg: configuration loaded from file
    Return:
        file_name: absolute path for the data file
    """
    s_data = {'cfg':cfg, 'data':list(data)}
    if not name:
        now = datetime.datetime.now()
        name = 'dataset-%02d-%02d-%02d-%02d-%02d.json' % ( now.month, now.day, now.hour, now.minute, now.second)


    file_name = os.path.join(data_dir,name)
    with open(file_name, 'w') as fp:
        fp.write(json.dumps(s_data, indent=3))
    return file_name

def loaddata(file_name):
    """
    Data will be loaded from json format
    
    Args:
        file_name: absolute path for the data file
    Return:
        cfg: configuration of the generated data
        data: data generated with ground truth
    """
    assert(os.path.isfile(file_name))
    s_data = json.load(open(file_name, 'r'))
    return s_data['cfg'], s_data['data']



def main():
    global args
    args = parser.parse_args()
    cfg = load_config(args.config)['data']
    data = generate_data(cfg)

    file_name = savedata(data, cfg)
    
def test(file_name):
    cfg,data = loaddata(file_name)
    print('Length of data:%d' %len(data))
    for i in range(len(data)):
        print(len(data[i]['clients']))


if __name__ == '__main__':
    #main()
    test('./dataset/synthetic/dataset-04-28-09-18-22.json')
