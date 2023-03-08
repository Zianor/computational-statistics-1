from deap import base, creator, tools
import random
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
import pandas as pd
from scipy.linalg import expm
from algorithm.utilities import graph

clusters = pd.read_csv('a1_data_clustered.csv')["cluster"]

def read_data():
    X = pd.read_csv('a1_data.csv') #names=['A','B','C','D','E','F','G','H','I','J','K']
    X = X.values
    return X

def sortnregress(X, alpha, fit_intercept):
    lasso = Lasso(alpha, fit_intercept=fit_intercept)

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


def create_individual(X, alpha_factor, use_cluster_inits, fit_intercept):
    if use_cluster_inits and random.random() < 0.5:
        cluster = random.choice(range(13))
        X_ = X[clusters==cluster]
        return sortnregress(X_, alpha=alpha_factor * random.random(), fit_intercept=fit_intercept)
    return sortnregress(X, alpha=alpha_factor * random.random(), fit_intercept=fit_intercept)


def fit_nodes(ind, X, fit_intercept):
    """Function to fit a DAG where the weight of the edges is unknown

    :param ind: individual, matrix with zeros and ones
    """
    edges = np.array(ind[0])
    edges_with_weights = np.zeros(edges.shape)
    intercepts = np.zeros((edges.shape[0],1))
    for node, incoming_edges in enumerate(edges.T):
        # flatten to 1d
        incoming_edges = incoming_edges.ravel()

        if sum(incoming_edges) > 0:
            edge_filter = np.argwhere(incoming_edges!=0).ravel()
            
            # Our X for the linear regression is the data coming from our incoming edges
            incoming_values = X[:, edge_filter]
            
            # Our y for the linear regression is the data of the current node
            node_values = X[:, node]

            lr = LinearRegression(fit_intercept=fit_intercept)
            lr.fit(incoming_values, node_values)
            
            edges_with_weights[edge_filter, node] = lr.coef_
            intercepts[node] = lr.intercept_
    return edges_with_weights, intercepts


def mse(X, W, intercepts):
    X_pred = intercepts + W.T @ X.T
    X_res = X.T - X_pred
    error_per_node = np.mean(X_res**2, axis=1)
    return np.mean(error_per_node)

def variance_of_residuals(X, W):
    X_pred = W.T @ X.T
    X_res = X.T - X_pred
    var = np.var(X_res, axis=1)
    return np.mean(var)

def evaluate(individual, X, fit_intercept):
    """Fitness function
    """

    if has_cycle(individual):
        return

    W, intercepts = fit_nodes(individual, X, fit_intercept)
    mean_mse = mse(X, W, intercepts)
    return mean_mse, np.count_nonzero(individual[0])

def mate(ind1, ind2, edge_addition_probability=0.7):
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
            if random.random() < edge_addition_probability:
                child1[i,j] = 1
                if has_cycle([child1]):  # better cycle removal?
                    child1[i,j] = 0

            if random.random() < edge_addition_probability:
                child2[i,j] = 1
                if has_cycle([child2]):  # better cycle removal?
                    child2[i,j] = 0            

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

    return ind, True
    
def has_cycle(ind) -> bool:
    """Check if the individual has a cycle
    """

    edges = np.array(ind[0])

    return np.trace(expm(np.multiply(edges, edges))) != edges.shape[0]
