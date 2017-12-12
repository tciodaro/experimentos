

import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys
import multiprocessing as mp


sys.path.append(os.getcwd() + "/../")

from Swarm import PSO
from Problems import Problem_Antenna
from sklearn.externals import joblib



"""
problem = Problem_Antenna.Problem()
#problem.load_test() # for testing
problem.load()
swarm_dim = problem.get_problem_dim()
swarm_min, swarm_max = problem.get_solution_limits()
ninit = 50
init_costs = np.zeros(ninit)
psoArgs = {
    'nswarm': 40,
    'swarm_dim': swarm_dim,
    'epochs': 1000,
    'chi': -1, #0.729,
    'rate_cog': 2.05,
    'rate_soc': 2.05,
    'K': 5,
    'Wup': 1.5,
    'Wlo': 0.5,
    'Wstep': 900,
    'swarm_max': swarm_max,
    'swarm_min': swarm_min,
    'speed_lim': -1,
    'trn_nproc': 4
}
best_pso = None
best_cost = 1e20
for iinit in range(ninit):
    print('Initialization: ', iinit)
    mypso = PSO.PSO(psoArgs)
    mypso.cost_func = problem.cost_function
    mypso.initialize()
    (solution_idx, sol_cost) = mypso.train()
    print("\tSolution found with cost: ", sol_cost)
    init_costs[iinit] = sol_cost
    if best_cost > sol_cost:
        best_cost = sol_cost
        best_pso = copy.deepcopy(mypso)
mypso = best_pso
opt_solution = mypso.swarm_best[-1]
"""

    
def train_parallel_pso(iinit):    
    problem = Problem_Antenna.Problem()
    #problem.load_test() # for testing
    problem.load()
    swarm_dim = problem.get_problem_dim()
    swarm_min, swarm_max = problem.get_solution_limits()
    psoArgs = {
        'nswarm': 40,
        'swarm_dim': swarm_dim,
        'epochs': 1000,
        'chi': -1, #0.729,
        'rate_cog': 2.05,
        'rate_soc': 2.05,
        'K': 5,
        'Wup': 1.5,
        'Wlo': 0.5,
        'Wstep': 900,
        'swarm_max': swarm_max,
        'swarm_min': swarm_min,
        'speed_lim': -1,
    }
    mypso = PSO.PSO(psoArgs)
    mypso.cost_func = problem.cost_function
    mypso.initialize()
    (solution_idx, sol_cost) = mypso.train()
    print("\tInit. %i: Solution found with cost: %.1f"%(iinit, sol_cost))
    return mypso


    
    
if __name__ == '__main__':    
    ## TRAIN
    nproc = 30
    ninit = 50
    pool = mp.Pool(processes=nproc)
    results = [pool.apply_async(train_parallel_pso, args=(i,)) for i in range(ninit)]
    results = [p.get() for p in results]
    scores = [obj.fitness[-1] for obj in results]
    idx = np.argmin(scores)
    mypso = results[idx]
    opt_solution = mypso.swarm_best[-1]
    print('Best result: ', scores[idx])
    print('Plotting')
    # Get problem
    problem = Problem_Antenna.Problem()
    #problem.load_test() # for testing
    problem.load()
    # PLOTS
    plt.figure(figsize=(20,4))
    plt.subplot(1,3,1)
    plt.plot(mypso.global_diversity, 'k', lw=2)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Swarm Diversity')
    plt.subplot(1,3,2)
    plt.plot(mypso.diversity, lw=2)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Dimension Diversity')
    plt.subplot(1,3,3)
    plt.plot(mypso.fitness)
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Swarm Best Fitness')
    plt.savefig("antenna_pso_epochs.png")


    # PLOT MAP
    plt.figure(figsize=(12,8))
    problem.plot_map()
    ax = plt.axis()
    # PLOT BASIC PSO SOLUTION
    problem.plot_solution(opt_solution, 'b')
    plt.axis(ax);
    plt.savefig('antenna_pso_map.png')


    problem.plot_swarm_evolution(mypso.swarm_best, 100, 'antenna_pso_evolution.gif')
    os.system("convert antenna_pso_evolution.gif -trim +repage -border 0  antenna_pso_evolution.gif")


    joblib.dump(mypso, 'pso_antenna_model.jbl',compress=9)



