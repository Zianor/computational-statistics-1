from deap import base, creator, tools
import random
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
import pandas as pd

def read_data():
    X = pd.read_csv('a1_data.csv') #names=['A','B','C','D','E','F','G','H','I','J','K']
    X = X.values
    return X

def sortnregress(X, alpha=0.01):
    # TODO: discuss difference to LR + LL
    lasso = Lasso(alpha)

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))
    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]
        lasso.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(lasso.coef_)
        # convert to 0-1
        weight[weight > 0] = 1
        
        W[covariates, target] = weight
    return W


def create_individual(X):
    # TODO better alpha 
    return sortnregress(X, alpha=random.random()/1)


def fit_nodes(ind, X):
    """Function to fit a DAG where the weight of the edges is unknown

    :param ind: individual, matrix with zeros and ones
    """
    edges = np.array(ind[0])
    edges_with_weights = np.zeros(edges.shape)
    for node, incoming_edges in enumerate(edges.T):
        # flatten to 1d
        incoming_edges = incoming_edges.ravel()

        if sum(incoming_edges) > 0:
            edge_filter = np.argwhere(incoming_edges!=0).ravel()
            
            # Our X for the linear regression is the data coming from our incoming edges
            incoming_values = X[:, edge_filter]
            
            # Our y for the linear regression is the data of the current node
            node_values = X[:, node]

            lr = LinearRegression(fit_intercept=False) # TODO do we fit an intercept?
            lr.fit(incoming_values, node_values)
            
            edges_with_weights[edge_filter, node] = lr.coef_
    return edges_with_weights


def mse(X, W):
    X_pred = W.T @ X.T
    X_res = X.T - X_pred
    error_per_node = np.mean(X_res**2, axis=1)
    # TODO: how to combine errors? depending on node?
    return np.mean(error_per_node)


def evaluate(individual, X):
    """Fitness function
    """
    W = fit_nodes(individual, X)
    # make sure the individual still fulfils the requirements of a DAG
    # TODO: normally returns two nodes
    # TODO: include number of edges in fitness calculation
    error = mse(X,W)
    return error,


def mate(ind1, ind2):
    """Mating function. Combines to individuals
    """
    ind1 = np.array(ind1[0])
    ind2 = np.array(ind2[0])

    child1 = np.zeros(ind1.shape)
    child2 = np.zeros(ind1.shape)

    for (i, j), _ in np.ndenumerate(ind1):
        # assuming as if 1/0 graph
        epsilon = 0.001
        if np.abs(ind1[i,j]) < epsilon and np.abs(ind2[i,j]) < epsilon:
            child1[i,j] = 0
            child2[i,j] = 0
        elif np.abs(ind1[i,j]) >= epsilon and np.abs(ind2[i,j]) >= epsilon:
            child1[i,j] = 1
            child2[i,j] = 1
        else:
            child1[i,j] = random.randint(0,1)
            child2[i,j] = random.randint(0,1)

    return child1.tolist(), child2.tolist()


def mutate(ind):
    """Mutation funciton
    """
    # create list of indices of non-zero elements
    ind = np.array(ind)
    ind = ind.ravel()
    non_zero = np.nonzero(ind)[0]
    
    # choose random index
    if non_zero.size > 0:
        index = random.choice(non_zero)
        # mutate
        ind[index] = 0

    # TODO more complex mutation, mutate 0 to n edges, adding edges?

    return ind,
