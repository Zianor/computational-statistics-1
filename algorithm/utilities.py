# visualize the matrix as a directed graph
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoLarsIC
import numpy as np

def graph(W):
    G = nx.DiGraph(W)
    G = nx.relabel_nodes(G, {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K'})
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='w', node_size=1000, font_size=20)
    plt.show()


def sortnregress_classic(X):
    LR = LinearRegression()
    LL = LassoLarsIC(criterion='bic')

    d = X.shape[1]
    W = np.zeros((d, d))
    increasing = np.argsort(np.var(X, axis=0))
    for k in range(1, d):
        covariates = increasing[:k]
        target = increasing[k]
        LR.fit(X[:, covariates], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, covariates] * weight, X[:, target].ravel())
        W[covariates, target] = LL.coef_ * weight
    return W
