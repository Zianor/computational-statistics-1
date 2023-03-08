from algorithm.instantiation import CausalDiscoveryGA
from algorithm.utilities import graph
import numpy as np
from algorithm.genetic_algorithm import evaluate, mse, sortnregress, read_data
import random
import optuna

X = read_data()

def objective(trial):
    alpha_factor = trial.suggest_float("alpha_factor", 0.01, 0.5, log=True)
    use_cluster_inits = trial.suggest_categorical("use_cluster_inits", [True, False])
    select_best = trial.suggest_categorical("select_best", [True, False])
    n_pop = trial.suggest_int("n_pop", 1, 100)
    n_gen = trial.suggest_int("n_gen", 2, 16, log=True)
    causalGA = CausalDiscoveryGA()
    causalGA.initialize_env(alpha_factor=alpha_factor, use_cluster_inits=use_cluster_inits,
                            n_pop=n_pop, n_gen=n_gen, select_best=select_best)
    best_individual = causalGA.start_ga()
    trial.set_user_attr("best_individual", best_individual[0][0].tolist())
    n_edges = int(np.sum(np.sum(best_individual[0][0] != 0, axis=1), axis=0))
    print(n_edges)
    trial.set_user_attr("n_edges", n_edges)
    return evaluate(best_individual[0], causalGA.X)[0]

def main():
    study = optuna.create_study(study_name="study_1", storage="sqlite:///hyperopt.db", load_if_exists=True, direction="minimize")
    study.optimize(objective, n_trials=20)
    best_individual = np.array(study.best_trial.user_attrs["best_individual"])
    graph(np.array(best_individual))
    # fit nodes for best individual
    print(f"MSE over all nodes: {evaluate([best_individual], X)}")
    print(f"MSE for sortnregress: {evaluate([sortnregress(X)], X)}")                            


if __name__ == "__main__":
    random.seed(23)

    main()
