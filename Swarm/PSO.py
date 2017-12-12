
import numpy as np
import time

"""
    Minimizer
"""
class PSO(object):
    def __init__(self, args):
        self.nswarm = -1
        self.swarm_dim = -1
        self.epochs = -1
        self.chi = -1
        self.rate_cog = -1
        self.rate_soc = -1
        self.K = -1
        self.Wup = -1
        self.Wlo = -1
        self.Wstep= -1
        self.swarm_max = -1
        self.swarm_min = -1
        self.speed_lim = -1
        if isinstance(args, PSO):
            args = args.__dict__
        self.__set_parameters(args)
        self.swarm = None
        self.speed = None
        self.swarm_best = None
        self.diversity = None
        self.fitness = None
        self.cost_func = None
        self.swarm_evolution = None        
    ######################################################################################
    def initialize(self, swarm = None, speed = None, seed = None):
        if self.cost_func is None:
            print('PSO: cost function not set')
            return False
        if self.swarm_dim == -1 or self.nswarm == -1:
            print('PSO: arguments not set correctly')
            return False
        # Do I have a pre-defined initialization?

        if swarm is not None:
            self.swarm = np.array(swarm)
            (self.nswarm, self.swarm_dim) = self.swarm.shape
        else:
            if seed is not None: np.random.seed(seed)
            self.swarm = np.random.rand(self.nswarm, self.swarm_dim)* \
                         (self.swarm_max - self.swarm_min) + self.swarm_min
        if speed is not None:
            self.speed = np.array(speed)
        else:
            if seed is not None: np.random.seed(seed+1)
            self.speed = np.random.rand(self.nswarm, self.swarm_dim)* \
                         (self.swarm_max - self.swarm_min) + self.swarm_min
        vmax = (self.swarm_max-self.swarm_min) / self.K if self.speed_lim == -1 else self.speed_lim
        self.speed[self.speed >  vmax] =  vmax
        self.speed[self.speed < -vmax] = -vmax
        # Training structures
        self.swarm_best = np.zeros((self.epochs + 1, self.swarm_dim)) * np.nan
        self.diversity = np.zeros((self.epochs, self.swarm_dim)) * np.nan
        self.global_diversity = np.zeros(self.epochs) * np.nan
        self.fitness = np.zeros(self.epochs) * np.nan
        self.swarm_evolution = np.zeros((self.epochs + 1, self.nswarm, self.swarm_dim)) * np.nan
    ######################################################################################
    def train(self, args = None):
        t0 = time.time()
        print('PSO Training started...',)
        # Particle memory, speed and Evaluate initial swarm
        fswarm = self.__cost()
        P = np.array(self.swarm)
        Pcost = np.array(fswarm)
        gbest = np.argmin(Pcost)
        gbest_cost = Pcost[gbest]
        # Inertia weight, constriction and speed limit
        self.chi = 1 if self.chi == -1 else self.chi
        wstep = (self.Wup - self.Wlo) / self.Wstep
        inertia = np.arange(self.Wup, self.Wup - self.epochs*wstep, -wstep)
        if self.Wstep < self.epochs: inertia[self.Wstep:] = inertia[self.Wstep]
        inertia = np.ones(self.epochs) if self.chi != 1 else inertia
        ## Random values used
        np.random.seed()
        R1 = np.random.rand(self.epochs,self.nswarm, self.swarm_dim)
        R2 = np.random.rand(self.epochs,self.nswarm, self.swarm_dim)
        ## SWARM EVOLUTION
        training_time = 0
        costfunc_time = 0
        self.swarm_evolution[0] = self.swarm
        self.swarm_best[0] = self.swarm[gbest]
        V = self.speed        
        vmax = (self.swarm_max-self.swarm_min) / self.K if self.speed_lim == -1 else self.speed_lim
        for it in range(self.epochs):
            t_train = time.time()
            ########################################## Update speed
            # Loop over particles
            for i in range(self.nswarm):
                W = inertia[it]
                V[i] = self.chi*(W*V[i] + self.rate_cog * R1[it][i] *(P[i]    - self.swarm[i]) +\
                                          self.rate_soc * R2[it][i] *(P[gbest]- self.swarm[i]))
            V[V >  vmax] =  vmax
            V[V < -vmax] = -vmax
            self.speed = V
            ########################################## Update swarm
            self.swarm = self.swarm + self.speed
            self.swarm[self.swarm > self.swarm_max] = self.swarm_max
            self.swarm[self.swarm < self.swarm_min] = self.swarm_min
            ########################################## Evaluate swarm            
            t_cost = time.time()
            fswarm = self.__cost()
            costfunc_time = costfunc_time + (time.time() - t_cost)
            
            idx = fswarm < Pcost
            P[idx] = self.swarm[idx] # update memory
            Pcost[idx] = fswarm[idx]
            gbest = np.argmin(Pcost)
            gbest_cost = Pcost[gbest]
            ########################################## Update training monitor
            self.fitness[it] = gbest_cost
            self.diversity[it] = np.std(self.swarm, axis=0)
            self.global_diversity[it] = self.diversity[it].std()
            self.swarm_best[it+1] = P[gbest]
            self.swarm_evolution[it+1] = self.swarm
            # Measure Time
            training_time = training_time + (time.time() - t_train)
            
        print('done (%.2f s)'%(time.time()-t0))
        print('\tTime epochs: %.2f s'%(training_time))
        print('\tTime cost F: %.2f s'%(costfunc_time))        
        return (gbest, gbest_cost)
    ######################################################################################
    def __cost(self):
        Z = np.array([self.cost_func(cs) for cs in self.swarm])
        Z = Z if Z.ndim == 1 else Z[:,0]
        return Z
    ######################################################################################
    def __set_parameters(self, args):
        argnames = ['nswarm','swarm_dim','epochs','chi','rate_cog',
                    'rate_soc','K','Wup','Wlo','Wstep',
                    'swarm_max', 'swarm_min', 'speed_lim',]
        argdefs  = [10, -1, 1000, 0.729, 2.05,
                    2.05, 10, 1.2, 0.1, 100,
                    1,-1,-1, 2]
        for name, value in zip(argnames, argdefs):
            self.__dict__[name] = args[name] if name in args.keys() else value


