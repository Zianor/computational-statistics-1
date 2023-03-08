from algorithm.instantiation import CausalDiscoveryGA
from algorithm.utilities import graph, sortnregress_classic
import numpy as np
from algorithm.genetic_algorithm import evaluate, read_data, sortnregress
import random
import optuna

X = read_data()

def objective(trial, return_solution=False):
    alpha_factor = trial.suggest_float("alpha_factor", 0.01, 0.5, log=True)
    use_cluster_inits = trial.suggest_categorical("use_cluster_inits", [True, False])
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    select_best = trial.suggest_categorical("select_best", [True, False])
    n_pop = trial.suggest_int("n_pop", 1, 100)
    n_gen = trial.suggest_int("n_gen", 2, 16, log=True)
    causalGA = CausalDiscoveryGA()
    causalGA.initialize_env(alpha_factor=alpha_factor, use_cluster_inits=use_cluster_inits,
                            n_pop=n_pop, n_gen=n_gen, fit_intercept=fit_intercept, select_best=select_best)
    best_individual = causalGA.start_ga()
    trial.set_user_attr("best_individual", best_individual[0][0].tolist())
    n_edges = int(np.sum(np.sum(best_individual[0][0] != 0, axis=1), axis=0))
    print(n_edges)
    trial.set_user_attr("n_edges", n_edges)
    if return_solution:
        return best_individual[0]
    return evaluate(best_individual[0], causalGA.X, fit_intercept)[0]

def hyperopt():
    study = optuna.create_study(study_name="study_1", storage="sqlite:///hyperopt.db", load_if_exists=True, direction="minimize")
    study.optimize(objective, n_trials=10)

def main():
    fit_intercept = True
    best_individual = objective(
        optuna.trial.FixedTrial(
            {
                "alpha_factor": 0.1,
                "use_cluster_inits": True,
                "fit_intercept": fit_intercept,
                "n_pop": 50,
                "n_gen": 8,
                "select_best": True,
                "edge_addition_probability": 0.7
            }
        ),
        return_solution=True,
    )
    graph(np.array(best_individual[0]))
    # fit nodes for best individual
    print(f"MSE over all nodes: {evaluate([best_individual[0]], X, fit_intercept=fit_intercept)}")
    print(f"MSE for classic sortnregress: {evaluate([sortnregress_classic(X)], X, fit_intercept=fit_intercept)}")
    print(f"MSE for lasso-regr. sortnregress: {evaluate([sortnregress(X, alpha=0.02, fit_intercept=fit_intercept)], X, fit_intercept=fit_intercept)}")

if __name__ == "__main__":
    random.seed(23)

    main()
