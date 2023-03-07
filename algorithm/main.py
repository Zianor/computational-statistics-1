from algorithm.instantiation import CausalDiscoveryGA
from algorithm.utilities import graph
import numpy as np
from algorithm.genetic_algorithm import evaluate, sortnregress
import random

def main():
    causalGA = CausalDiscoveryGA()
    causalGA.initialize_env()
    best_individual = causalGA.start_ga()
    print(best_individual[0][0])
    graph(np.array(best_individual[0][0]))
    # fit nodes for best individual
    print(f"MSE over all nodes: {evaluate(best_individual[0], causalGA.X)}")
    print(f"MSE for sortnregress: {evaluate([sortnregress(causalGA.X)], causalGA.X)}")                            


if __name__ == "__main__":
    random.seed(23)

    main()
