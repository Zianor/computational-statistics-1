from algorithm.instantiation import CausalDiscoveryGA
from algorithm.utilities import graph
import numpy as np

def main():
    causalGA = CausalDiscoveryGA()
    causalGA.initialize_env()
    best_individual = causalGA.start_ga()
    print(best_individual[0][0])
    graph(np.array(best_individual[0][0]))


if __name__ == "__main__":
    main()
