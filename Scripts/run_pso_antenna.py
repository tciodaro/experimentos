

import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys

sys.path.append(os.getcwd() + "/../")

from Swarm import Problem_Antenna, PSO


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
    'K': 10.0, # 20
    'Wup': 1.2,
    'Wlo': 0.5,
    'Wstep': 800,
    'swarm_max': swarm_max,
    'swarm_min': swarm_min,
    'speed_lim': -1,
    'trn_nproc': 2
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



# PLOTS
plt.figure(figsize=(15,4))
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


problem.plot_swarm_evolution(mypso.swarm_best, 50, 'antenna_pso_evolution.gif')

