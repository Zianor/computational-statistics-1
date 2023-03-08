from deap import base, creator, tools
from algorithm.genetic_algorithm import read_data, create_individual, evaluate, mate, mutate, has_cycle
import random
from tqdm import tqdm
import numpy as np

class CausalDiscoveryGA:
    """Class to instantiate the genetic algorithm.
    """

    def __init__(self):
        """Initialize the genetic algorithm.
        """
        self.pop = None
        self.toolbox = None
        self.IND_SIZE = 1
        self.X = read_data()
        
    def initialize_env(self, alpha_exponent, use_cluster_inits, n_pop, n_gen, fit_intercept, edge_addition_probability=0.7,
                       select_best=False):
        """Initialize the GA environment.
        """
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.select_best = select_best
        self.fit_intercept = fit_intercept

        # minimize fitness  
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))

        # 2d list to represent DAG
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # creating population
        toolbox.register("attribute", create_individual, self.X, alpha_exponent, use_cluster_inits, fit_intercept)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attribute, n=self.IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # mate, mutate, select and evaluate functions
        toolbox.register("mate", mate, edge_addition_probability=edge_addition_probability)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate, X=self.X, fit_intercept=fit_intercept)

        self.toolbox = toolbox


    def start_ga(self):
        """Start the genetic algorithm.
        """

        # create initial population
        pop = self.toolbox.population(n=self.n_pop)
        CXPB, MUTPB, NGEN = 0.5, 0.2, self.n_gen

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        fitnesses = filter(lambda x: x is not None, fitnesses)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in tqdm(range(NGEN)):
            best_pop = tools.selBest(pop, 10)
            evaluated = [evaluate(best, self.X, self.fit_intercept) for best in best_pop]
            evaluated = [best for best in evaluated if best is not None]  # in case we have invalid DAGs
            print(f"Avg for 10 best individuals in generation {g}: {np.mean(evaluated)}")
            # print number of edges of each individual
            print(f"n_edges for 10 best individuals in generation {g}: {[np.sum(np.sum(best[0])) for best in best_pop]}")

            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            fitnesses = filter(lambda x: x is not None, fitnesses)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if not self.select_best:
                # The population is entirely replaced by the offspring
                pop[:] = offspring
            else:
                # select the best from population and offspring
                pop[:] = self.toolbox.select(pop + offspring, len(pop))

        # select best individual
        best_individual = tools.selBest(pop, 1)

        return best_individual