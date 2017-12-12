


import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.getcwd() + "\\..\\")

from Problems.Problem_Gaussian import Problem
from Swarm import PSO

ngauss = 20
problem = Problem(ngauss, 10)
plt.figure(figsize=(10,6))
problem.plot(colorbar=True)
plt.title('Cost Function', fontsize=18, fontweight='bold');



swarm_dim = problem.get_problem_dim()
swarm_min, swarm_max = problem.get_solution_limits()
psoArgs = {
    'nswarm': 20,
    'swarm_dim': swarm_dim,
    'epochs': 20,
    'chi': 0.0, #0.729
    'rate_cog': 2.05,
    'rate_soc': 2.05,
    'K': 5.0,
    'Wup': 1.5,
    'Wlo': 0.5,
    'Wstep': 20,
    'swarm_max': swarm_max,
    'swarm_min': swarm_min,
    'speed_lim': -1
}

mypso = PSO.PSO(psoArgs)
mypso.cost_func = problem.cost_function
seed = 40
mypso.initialize(seed=seed)
# Initial Swarm
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
problem.plot(mypso.swarm)
plt.title('Initial Swarm', fontsize=18, fontweight='bold');

(solution_idx, sol_cost) = mypso.train()
opt = problem.get_optimal()

print('Analytical Optimal: %.2f, at position (%.2f, %.2f)'%(opt[1], opt[0][0], opt[0][1]))
print("Solution found with cost: %.2f, at position (%.2f, %.2f)"%(sol_cost, mypso.swarm_best[-1][0],
                                                                  mypso.swarm_best[-1][1]))


plt.subplot(2,2,2)
problem.plot(mypso.swarm)
plt.plot(opt[0][0], opt[0][1], 'sr', ms=8)
plt.title('Final Swarm Position', fontsize=18, fontweight='bold')

opt_solution = mypso.swarm_best[-1]
plt.subplot(2,2,3)
problem.plot(mypso.swarm_best, fmt='-ob')
plt.plot(opt[0][0], opt[0][1], 'sr', ms=8)
plt.title('Swarm Optimal Evolution', fontsize=18, fontweight='bold')

plt.subplot(2,2,4)
plt.plot(opt[0][0], opt[0][1], 'sr', ms=8)
problem.plot(mypso.swarm_best[-1], 'sg')
plt.title('Final Optimal Found', fontsize=18, fontweight='bold');

plt.savefig("../Static/pso_gaussian_example_4.png")

