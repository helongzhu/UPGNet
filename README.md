
## Requirements

This code is implemented in Python 3.9, and relies on the following packages:  
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.8.1
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) >= 1.7.0
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) >= 1.2.4
- [Numpy](https://numpy.org/install/) >= 1.20.2
- [Seaborn](https://seaborn.pydata.org/) >= 0.11.1  


## Usage

###  Training individual models
To replicate our experiments:  
1. Run ``python main.py --dataset cora --mechanism mbm --model sage --x_eps 0.01 --x_steps 8  --learning_rate 0.01 --weight_decay 0.01 --dropout 0.75 -s 12345 -r 1 -o ./test``

```
python main.py [OPTIONS...]

  -dataset           name of the dataset (default: cora)
  --mechanism        feature perturbation mechanism (mbm, mpm) 
  --model            backbone GNN model (gcn, sage, gat)
  --x_eps            privacy budget (0.01, 0.1, 1, 2, 3, np.inf)
  --x_step           the step parameter of HOA
  --learning-rate    learning rate
  --weight-decay     weight decay
  --dropout          dropout rate
  -s                 initial random seed
  -r                 number of times the experiment is repeated (default: 1)
  -o                 directory to store the results
