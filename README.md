# gcn-flc

graph convolution network on incapacitated facility location problem  

## Data preparation

### Install Gurobi

Register and download `Gurobi` from [here](http://www.gurobi.com/registration/download-reg).

Then export environment variables.

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/absolute/path/gurobi811/linux64/lib
```

Most importantly, you need a liscense from gurobi. You can get one from [here](http://www.gurobi.com/downloads/licenses/license-center). Also, you need to connect hopkins network or use `Pulse Secure`

### Install self-defined package

```shell
    pip install -e .
```

on the root folder which has `setup.py`.

###

Generated data can be downloaded [here](https://drive.google.com/open?id=1A681BhENhVmR-PEZiLzN5iJ5paz2-40K)

## Examples

### Graph generation

An example of graph
![image](https://github.com/YufanHe/gcn-flc/blob/dev_pengfei/media/graph_ex.png)
An example of solution
![image](https://github.com/YufanHe/gcn-flc/blob/dev_pengfei/media/solution_ex.png)

## Training

### Install requirement

```shell
conda install -c conda-forge tqdm tensorboardx protobuf
conda install tensorflow
conda install pytorch=0.4.1 cuda90 -c pytorch
git clone https://github.com/sksq96/pytorch-summary
rm -r pytorch-summary/.git
```