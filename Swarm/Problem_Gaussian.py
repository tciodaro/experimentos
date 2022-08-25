import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import stats


class Problem(object):
    def __init__(self, N = 4, seed = 81473):
        if seed == -1:
            seed = np.random.randint(1, 100000,1)[0];
        np.random.seed(seed);
        self.sigma = np.random.rand(N,2)/2 + 0.5   
        np.random.seed(seed+1);
        self.amp = np.random.rand(N) * 1000;
        np.random.seed(seed+2);
        self.avg = np.random.rand(N,2)*10 - 5;
        self.xmin = -5
        self.xmax =  5
        self.ymin = -5
        self.ymax =  5        
    
    def cost_function(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = np.array([X.tolist()])
        N = self.sigma.shape[0]
        Z = np.zeros(X.shape[0])
        for i in range(N):
            A = self.amp[i]
            M = self.avg[i]
            S = self.sigma[i]
            # Z = Z - mlab.bivariate_normal(X[:,0], X[:,1], S[0], S[1], M[0], M[1], 0)
            rv = stats.multivariate_normal([ M[0], M[1]], [[S[0], 0], [0, S[1]]])
            Z = Z - rv.pdf(np.dstack((X[:,0], X[:,1])))
        return Z
    
    def plot(self, candidates = None, fmt = 'ok', fig = None, colorbar=None):
        if fig is not None:
            plt.figure(fig.number)
        # plot contours of cost function
        X = np.linspace(self.xmin, self.xmax,101)
        Y = np.linspace(self.ymin, self.ymax,101)
        X,Y = np.meshgrid(X,Y)
        domain = np.array([X.reshape(1,-1)[0], Y.reshape(1,-1)[0]]).T
        Z = self.cost_function(domain)
        cs = plt.contour(X,Y,Z.reshape(101,101), 50)
        # plot candidates
        if candidates is not None:
            candidates = np.array(candidates)
            if candidates.ndim == 1:
                candidates = np.array([candidates.tolist()])
            plt.plot(candidates[:,0], candidates[:,1], fmt)
            for i, xy in enumerate(candidates):
                plt.gca().annotate('%i' % i, xy=xy, textcoords='data')
        plt.grid(True)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        if colorbar is not None:
            plt.colorbar()   
        
    def plot_swarm_evolution(self, candidates, best, savefile = 'swarm_evolution.gif'):
        fig = plt.figure(figsize=(8,6))
        # Generate grid for plotting
        X = np.linspace(self.xmin, self.xmax,101)
        Y = np.linspace(self.ymin, self.ymax,101)
        X,Y = np.meshgrid(X,Y)
        domain = np.array([X.reshape(1,-1)[0], Y.reshape(1,-1)[0]]).T
        Z = self.cost_function(domain)
        candidates = np.array(candidates)
        best = np.array(best)
        
        def animate(i):
            plt.figure(fig.number)
            plt.cla()
            ax = plt.axes(xlim=(self.xmin, self.xmax), ylim=(self.ymin, self.ymax))
            cs = plt.contour(X,Y,Z.reshape(101,101), 50)
            # plot candidates
            plt.plot(candidates[i,:,0], candidates[i,:,1], 'ok')
            # Plot best
            plt.plot(best[i,0], best[i,1], 'sr')
            plt.text(-5, -6, 'Iteration %i'%i)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])            
            
        anim = animation.FuncAnimation(fig, animate, frames=len(candidates))
        anim.save(savefile, writer='imagemagick', fps=4);    
        plt.close()
        
    def get_problem_dim(self):
        return self.avg.shape[1]
    
    def get_solution_limits(self):
        return (self.xmin, self.xmax)
    
    def get_optimal(self):
        X = np.linspace(self.xmin, self.xmax,101)
        Y = np.linspace(self.ymin, self.ymax,101)
        X,Y = np.meshgrid(X,Y)
        domain = np.array([X.reshape(1,-1)[0], Y.reshape(1,-1)[0]]).T
        Z = self.cost_function(domain)
        idx = np.argmin(Z)
        return ((domain[idx,0], domain[idx,1]), Z[idx])

# END OF FILE

