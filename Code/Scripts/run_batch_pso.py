
import copy
import numpy as np
import matplotlib.pyplot as plt
from CPE730 import Problem, BasicPSO
from sklearn.externals import joblib

savefile = 'pso_result.jbl'
savedir = "./Results/PSO_Chi_K5_S40_N1000"
ninit = 10

## LOAD DATA
print 'Starting problem'
problem = Problem.Problem()
#problem.load_test() # For testing
problem.load()
print 'Data loaded'

## PSO TRAINING
swarm_dim = problem.get_problem_dim()
swarm_min, swarm_max = problem.get_solution_limits()
psoArgs = {
    'nswarm': 40,
    'swarm_dim': swarm_dim,
    'epochs': 1000,
    'chi': 0.729,
    'rate_cog': 2.05,
    'rate_soc': 2.05,
    'K': 5.0,
    'Wup': 1.2,
    'Wlo': 0.2,
    'Wstep': 1000,
    'swarm_max': swarm_max,
    'swarm_min': swarm_min,
    'speed_lim': -1,
    'trn_nproc': 6
}

print 'Starting PSO Training'

best_pso = None
best_cost = 1e20
init_costs = np.zeros(ninit) * np.nan
for iinit in range(ninit):
    print '\tInitialization ', iinit
    mypso = BasicPSO.PSO(psoArgs)
    mypso.cost_func = Problem.cost_function
    mypso.initialize()
    (solution_idx, sol_cost) = mypso.train(problem.get_info())
    print "\t\tSolution found with cost: ", sol_cost
    init_costs[iinit] = sol_cost
    if best_cost > sol_cost:
        best_cost = sol_cost
        best_pso = copy.deepcopy(mypso)

mypso = best_pso
opt_solution = mypso.swarm_best[-1]

## PLOT TRAINING MONITORING
plt.ioff()

plt.figure()
plt.bar(np.arange(1,ninit+1), init_costs, color='k')
plt.grid(True)
plt.xlabel('Cost Value')
plt.ylabel('Initializations')
plt.savefig(savedir + "/costs_initializations.png")
plt.close()

plt.figure()
plt.plot(mypso.global_diversity, 'k', lw=2)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Swarm Diversity')
plt.savefig(savedir + "/swarm_diversity.png")
plt.close()

plt.figure()
plt.plot(mypso.diversity, lw=2)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Dimension Diversity')
plt.savefig(savedir + "/swarm_dim_diversity.png")
plt.close()

plt.figure()
plt.plot(mypso.fitness)
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Swarm Best Fitness')
plt.savefig(savedir + "/swarm_fitness.png")
plt.close()


## PLOT SOLUTION ON MAP
plt.figure(figsize=(12,8))
problem.plot_map()
ax = plt.axis()
problem.plot_solution(opt_solution, 'b')
plt.axis(ax);
plt.savefig(savedir + "/solution_map.png")
plt.close()


try:
    problem.plot_swarm_evolution(mypso.swarm_best, 'swarm_evolution.gif')
except Exception as err:
    print 'You do not have ImageMagick installed to create gifs. Aborting.'
    pass

print 'Done!'

if savefile is not None:
    savefile = savedir + '/' + savefile
    joblib.dump({'pso':mypso, 'problem':problem, 'init_costs':init_costs}, savefile, compress=9)


