# gcn-flc

graph convolution network on incapacitated facility location problem  

## Data preparation

Register and download `Gurobi` from [here](http://www.gurobi.com/registration/download-reg).

Then export environment variables.

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/absolute/path/gurobi811/linux64/lib
```

Most importantly, you need a liscense from gurobi. You can get one from [here](http://www.gurobi.com/downloads/licenses/license-center). Also, you need to connect hopkins network or use `Pulse Secure`

## Graph generation

An example of graph
![image](https://github.com/YufanHe/gcn-flc/blob/dev_pengfei/media/graph_ex.png)