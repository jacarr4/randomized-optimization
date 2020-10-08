import argparse
import matplotlib
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import numpy as np

# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):
    
    # Initialize counter
    fitness = 0
    
    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
            
            # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) \
                and (state[j] != state[i] + (j - i)) \
                and (state[j] != state[i] - (j - i)):
                
                # If no attacks, then increment counter
                fitness += 1

    return fitness

# note: genetic alg should excel at four peaks,
#       while RHC and annealing are likely to get stuck in the local optima with bigger basins (all 1s or all 0s).
#       annealing and RHC will perform well on onemax because there is only one optimum and it's global (wide basin).
#       MIMIC will perform best on kcolor, per Charles Isbell's paper

def get_score(alg, problem, params, num_seeds = 30):
    fitnesses = []
    fitness_curves = []
    for random_seed in range(1,num_seeds+1):
        params.update({'random_state': random_seed, 'curve': True})
        state, fitness, curve = alg(**params)
        fitnesses.append(fitness)
        fitness_curves.append(curve)
        # print(curve)

    avg_fitness_curve = [ np.mean( [ c[i] for c in fitness_curves ] ) for i in range(min([len(c) for c in fitness_curves])) ]
    return np.mean(fitnesses), avg_fitness_curve

def get_hyperparam_score(alg, problem, params):
    fitness, curve = get_score(alg, problem, params, 20)
    return fitness

SIZE = 20

def find_best_hyperparameter(alg, fitness, param_name, possible_values, show_graph = True):
    best_avg_score = 0
    best_val = None
    scores = []
    for v in possible_values:
        problem = mlrose.DiscreteOpt(length = SIZE, fitness_fn = fitness, maximize = True)
        params = {param_name: v, 'problem': problem}
        score = get_hyperparam_score(alg, problem, params)
        scores.append(score)
        if score > best_avg_score:
            best_avg_score = score
            best_val = v
    
    # plot it
    if show_graph:
        plt.plot(possible_values, scores)
        plt.xlabel( f'{param_name}' )
        plt.ylabel( 'Average Fitness Score' )
        plt.suptitle( f'{param_name}' )
        plt.show()

    return best_val        

def optimize_hyperparams(title, alg, fitness, params):
    hyperparams = {}
    for param_name, (possible_values, show_graph) in params.items():
        print( 'optimizing hyperparameter %s' % param_name )
        hyperparams[param_name] = find_best_hyperparameter(alg, fitness, param_name, possible_values, show_graph)
        print( 'optimal value found: %s' % hyperparams[param_name] )
    
    print('parameters optimized. plotting learning curve with optimal params.')

    problem = mlrose.DiscreteOpt(length = SIZE, fitness_fn = fitness, maximize = True)
    temp_params = hyperparams.copy()
    temp_params.update({'problem': problem})
    score, curve = get_score(alg, problem, temp_params)
    print('Score:', score)
    plt.plot( curve )
    plt.xlabel( 'Iterations' )
    plt.ylabel( 'Average fitness score' )
    plt.suptitle( title )
    plt.show()
    
    return hyperparams

def optimize_hill_climbing_hyperparams(fitness):
    params = { 'max_attempts': ( [ 100 ], False ),
               'max_iters': ( [ 100 ], False ),
               'restarts': ( [ 20 ], False ) }
    return optimize_hyperparams( 'Random Hill Climbing', mlrose.random_hill_climb, fitness, params )

def optimize_simulated_annealing_hyperparams(fitness):
    params = { 'max_attempts': ( [ 100 ], False ),
               'max_iters': ( [ 1000 ], False ) }
    return optimize_hyperparams( 'Simulated Annualing', mlrose.simulated_annealing, fitness, params )

def optimize_genetic_alg_hyperparams(fitness):
    params = { 'max_attempts': ( [ 100 ], False ),
               'pop_size': ( [ 100 + 20*i for i in range(11) ], True ),
               'pop_breed_percent': ( [ 0.7 + 0.01*i for i in range(11) ], True ),
               'elite_dreg_ratio': ( [ 0.95 + 0.01*i for i in range(5) ], True ),
               'mutation_prob': ( [ 0.2 * i for i in range(1, 5) ], True ) }
    return optimize_hyperparams( 'Genetic Alg', mlrose.genetic_alg, fitness, params )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--learner', action = 'store', dest = 'learner', required = True )
    parser.add_argument( '--problem', action = 'store', dest = 'problem', required = True )
    # parser.add_argument( '--size', action = 'store', dest = 'size', required = False, default = 8)
    args = parser.parse_args()

    # size = int(args.size)

    if args.problem == 'queens':
        fitness = mlrose.CustomFitness(queens_max)
        problem = mlrose.DiscreteOpt(length = SIZE, fitness_fn = fitness, maximize = True, max_val = size)
    elif args.problem == 'onemax':
        fitness = mlrose.OneMax()
        problem = mlrose.DiscreteOpt(length = SIZE, fitness_fn = fitness, maximize = True)
    elif args.problem == 'four_peaks':
        fitness = mlrose.FourPeaks(t_pct = 0.15)
        problem = mlrose.DiscreteOpt(length = SIZE, fitness_fn = fitness, maximize = True)
    elif args.problem == 'kcolor':
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        fitness = mlrose.MaxKColor(edges)
        problem = mlrose.DiscreteOpt(length = 5, fitness_fn = fitness)
        init_state = np.array([0, 1, 0, 1, 1])
    else:
        raise RuntimeError("Invalid problem argument")

    if args.learner == 'hill_climbing':
        hyperparams = optimize_hill_climbing_hyperparams(fitness)
        print( 'found hyperparams:', hyperparams )
    if args.learner == 'simulated_annealing':
        hyperparams = optimize_simulated_annealing_hyperparams(fitness)
        print( 'found hyperparams:', hyperparams )
    if args.learner == 'genetic_alg':
        hyperparams = optimize_genetic_alg_hyperparams(fitness)
        print( 'found hyperparams:', hyperparams )
        # avg_best_score, avg_fitness_curve = get_score()

    # best_fitnesses = []
    # fitness_curves = []

    # for random_seed in range(30):
    #     if args.learner == 'simulated_annealing':
    #         # Define decay schedule
    #         schedule = mlrose.ExpDecay()
    #         best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 100, max_iters = 1000, curve = True, random_state = random_seed)
    #     elif args.learner == 'hill_climbing':
    #         best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(problem, restarts = 100, max_attempts = 100, max_iters = 100, curve = True, random_state = random_seed)
    #     elif args.learner == 'genetic_alg':
    #         best_state, best_fitness, fitness_curve = mlrose.genetic_alg(problem, max_attempts = 100, curve = True)
    #     elif args.learner == 'mimic':
    #         best_state, best_fitness, fitness_curve = mlrose.mimic(problem, curve = True, random_state = random_seed)
    #     else:
    #         raise ValueError('Invalid learner argument')

    #     best_fitnesses.append(best_fitness)
    #     fitness_curves.append(fitness_curve)
    #     print('The curve is: ', fitness_curve)
    #     print(len( fitness_curve ))

    # avg_best_fitness = np.mean( best_fitnesses )
    # avg_fitness_curve = [ np.mean( [ c[i] for c in fitness_curves ] ) for i in range(100) ]
    # plt.plot( avg_fitness_curve )
    # plt.xlabel( 'Iterations' )
    # plt.ylabel( 'Average fitness score' )
    # plt.suptitle( args.learner )
    # plt.show()