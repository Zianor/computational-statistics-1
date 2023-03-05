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
    # TODO: check difference to LR + LL
    lasso = Lasso(alpha)

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))
    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]
        lasso.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(lasso.coef_)
        W[covariates, target] = weight
    return W

def create_individual(X):
    # TODO better alpha 
    return sortnregress(X, alpha=random.random()/10)


def fit_nodes(edges):
    """Function to fit a DAG where the weight of the edges is unknown

    :param edges: edge matrix with zeros and ones
    """
    edges_with_weights = np.zeros(edges.shape)
    for node, incoming_edges in enumerate(edges.T):
        lr = LinearRegression()
        if sum(incoming_edges) > 0:
            incoming_values = X[:, incoming_edges!=0]
            lr.fit(X[:, incoming_edges!=0], X[:, node].ravel())
            edges_with_weights[incoming_edges!=0, node] = lr.coef_
    return edges_with_weights


def mse(X, W):
    X_pred = W.T @ X.T
    X_res = X.T - X_pred
    return np.mean(X_res**2, axis=1)


def evaluate(individual):
    """Fitness function
    """
    # TODO: implement fitness function
    # make sure the individual still fulfils the requirements of a DAG
    return sum(individual),


def mate(ind1, ind2):
    """Mating function. Combines to individuals
    """
    ind1 = np.array(ind1)
    ind2 = np.array(ind2)

    child1 = np.zeros(ind1.shape)
    child2 = np.zeros(ind1.shape)

    for (i, j), _ in np.ndenumerate(ind1):
        # assuming as if 1/0 graph
        # TODO check for values close to 0
        if ind1[i,j] == 0 and ind2[i,j] == 0:
            child1[i,j] = 0
            child2[i,j] = 0
        elif ind1[i,j] != 0 and ind2[i,j] != 0:
            child1[i,j] = 1
            child2[i,j] = 1
        else:
            child1[i,j] = random.randint()
            child2[i,j] = random.randint()

    # TODO: regression for values

    return child1, child2


def mutate(ind):
    """Mutation funciton
    """
    # create list of indices of non-zero elements
    ind = np.array(ind)
    ind = ind.ravel()
    non_zero = np.nonzero(ind)[0]
    
    # choose random index
    index = random.choice(non_zero)

    # mutate
    ind[index] = 0

    # TODO more complex mutation, mutate 0 to n edges, adding edges?

    return ind,
