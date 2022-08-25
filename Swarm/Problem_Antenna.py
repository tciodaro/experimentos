from matplotlib.patches import Ellipse
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas
import numpy as np
import time
import os

import theano

class Problem(object):
    def __init__(self, datadir):
        self.G1ref = np.array([107, 579], 'f')
        self.G24ref = np.array([ 727, 40], 'f')
        self.df = {}
        self.dirname = datadir
        self.cost_func_params = {}
        # Theano function
        self.t_xsegs = theano.tensor.vector('xsegs')
        self.t_ysegs = theano.tensor.vector('ysegs')
        self.t_xref = theano.tensor.dscalar('xref')
        self.t_yref = theano.tensor.dscalar('yref')
        self.t_radius = theano.tensor.dscalar('radius')
        self.t_from_x = theano.tensor.dscalar('from_x')
        self.t_from_y = theano.tensor.dscalar('from_y')
        self.t_to_x = theano.tensor.dscalar('to_x')
        self.t_to_y = theano.tensor.dscalar('to_y')
        self.t_frac = theano.tensor.dscalar('frac')
        self.__theano_function = None
        self.__prepare_theano_function()
    ##################################################################################################
    def load_test(self):
        self.load(self.dirname+'/DadosFluxoTransito_Teste.xlsx')
    ##################################################################################################
    def load(self, datafile = None):
        datafile = datafile if datafile is not None else self.dirname + '/DadosFluxoTransito.xlsx'
        try:
            wb = pandas.ExcelFile(datafile)
        except:
            print('Could not open data file "'+datafile+'"')
            raise
        self.df['geo'] = wb.parse('Geo', index_col='Geo')
        self.df['seg'] = wb.parse('Segments', index_col = 'Segment')
        self.df['ant'] = wb.parse('Antennas', index_col = 'Type')
        self.df['sol'] = wb.parse('Solution', index_col = 'Segment')

        self.xFactor = np.abs((self.G24ref[0] - self.G1ref[0]) / (self.df['geo'].X[24]-self.df['geo'].X[1]))
        self.yFactor = np.abs((self.G24ref[1] - self.G1ref[1]) / (self.df['geo'].Y[24]-self.df['geo'].Y[1]))
        # Create subsegments for each segment
        nbins = 100
        col_names = ['Segment','Subsegment','X','Y','Traffic','nbins']
        subsegs = pandas.DataFrame(index = range(nbins * self.df['seg'].shape[0]), columns=col_names)
        ibeg = 0
        subsegs['nbins'] = nbins
        for row in self.df['seg'].iterrows():
            seg = row[0]
            gTo = self.df['geo'].loc[self.df['seg'].GeoTo[seg],['X','Y']]
            gFrom = self.df['geo'].loc[self.df['seg'].GeoFrom[seg], ['X','Y']]
            xbins = np.linspace(gTo[0], gFrom[0], nbins)
            ybins = np.linspace(gTo[1], gFrom[1], nbins)
            # Subsegments
            subsegs.loc[ibeg:nbins + ibeg - 1, 'Segment'] = seg
            subsegs.loc[ibeg:nbins + ibeg - 1, 'Subsegment'] = np.arange(nbins)
            subsegs.loc[ibeg:nbins + ibeg - 1, 'X'] = xbins
            subsegs.loc[ibeg:nbins + ibeg - 1, 'Y'] = ybins
            subsegs.loc[ibeg:nbins + ibeg - 1, 'Traffic'] = float(row[1].Flow) / nbins
            ibeg = nbins + ibeg
        self.df['subsegs'] = subsegs
        # Prepare cost function parameters
        df = self.df
        self.total_traffic = df['subsegs'].Traffic.sum()
        total_segs = df['seg'].shape[0]
        self.geo_to = np.array([df['geo'].loc[df['seg'].GeoTo[seg],['X','Y']] for seg in range(1, total_segs+1)])
        self.geo_from = np.array([df['geo'].loc[df['seg'].GeoFrom[seg],['X','Y']] for seg in range(1, total_segs+1)])
        
    ##################################################################################################
    def get_problem_dim(self):
        return self.df['seg'].shape[0]
    ##################################################################################################
    def get_solution_limits(self):
        return (0, self.df['ant'].index.max() + 0.99999)
    ##################################################################################################
    def get_info(self):
        return self.df
    ##################################################################################################
    def geo_plannar_approx(self,lat1, long1, lat2, long2):
        x = (long2-long1)*np.cos(lat1)*np.pi/180
        y = (lat2-lat1)*np.pi/180
        x = x * 10.97912711
        y = y * 10.97912711
        d = np.sqrt(x**2 + y**2)
        return (d, x, y)
    ##################################################################################################
    def plot_circle_on_map(self,X,Y,R,marker='o', color='k'):
        xmap = X * self.xFactor + self.G1ref[0]
        ymap = self.G1ref[1] - Y * self.yFactor
        radius = R*self.xFactor
        xR = R*self.xFactor
        yR = R*self.yFactor
        plt.plot(xmap, ymap, marker+color, ms = 5)
        #plt_circle = plt.Circle([xmap, ymap], radius=radius, color=color,fill=False)
        #plt.gca().add_artist(plt_circle)
        plt_ellipse= Ellipse(xy=[xmap,ymap], width=xR*2, height=yR*2,color=color, fill=False)
        plt.gca().add_artist(plt_ellipse)
    ##################################################################################################
    def plot_solution(self,solution, color='k'):
        for seg, value in enumerate(solution):
            # Get antenna type: it defines the radius
            ant_type = int(value)
            radius = self.df['ant'].Radius[ant_type]
            # Get antenna position
            frac = value - int(value)
            gTo = self.df['geo'].loc[self.df['seg'].GeoTo[seg+1],['X','Y']]
            gFrom = self.df['geo'].loc[self.df['seg'].GeoFrom[seg+1], ['X','Y']]
            x = gTo.X + (gFrom.X - gTo.X) * frac
            y = gTo.Y + (gFrom.Y - gTo.Y) * frac
            self.plot_circle_on_map(x, y, radius,color=color)
    ##################################################################################################
    def plot_on_map(self,X, Y, fmt = 'xr'):
        xmap = X * self.xFactor + self.G1ref[0]
        ymap = self.G1ref[1] - Y * self.yFactor
        plt.plot(xmap, ymap, fmt)
    ##################################################################################################
    def plot_map(self):
        filename = self.dirname + '/MapaRio.png'
        img = plt.imread(filename)
        plt.imshow(img)
    ##################################################################################################
    def plot_swarm_evolution(self,candidates, savefile = 'swarm_evolution.gif'):
        def animate(nframe):
            plt.cla()
            self.plot_map()
            ax = plt.axis()
            # PLOT BASIC PSO SOLUTION
            self.plot_solution(candidates[nframe], 'b')
            plt.text(850, 500, 'Iteration %i'%nframe)
            plt.axis(ax);

        fig = plt.figure(figsize=(6,8))
        anim = animation.FuncAnimation(fig, animate, frames=len(candidates))
        anim.save(savefile, writer='imagemagick', fps=4);

    ##################################################################################################
    def __prepare_theano_function(self):
        self.t_xref = self.t_to_x + (self.t_from_x - self.t_to_x) * self.t_frac # The center of the radius
        self.t_yref = self.t_to_y + (self.t_from_y - self.t_to_y) * self.t_frac
        r2 = theano.tensor.power(self.t_xsegs - self.t_xref, 2) + theano.tensor.power(self.t_ysegs - self.t_yref, 2)
        is_inside = r2 < self.t_radius**2
        self.__theano_function = theano.function(inputs=[self.t_xsegs, self.t_ysegs, self.t_to_x, self.t_to_y,
                                                         self.t_from_x, self.t_from_y, self.t_frac, self.t_radius],
                                                 outputs=[is_inside])    

    ##################################################################################################
    ## COST FUNCTION
    def cost_function(self, candidate):
        # Antenna and quality of service cost
        cost_ant = 0
        cost_qos = 0
        covered_segs = np.array([False]*self.df['subsegs'].shape[0])
        subsegs = [self.df['subsegs'].X, self.df['subsegs'].Y]
        for seg, sol in enumerate(candidate):
            radius = self.df['ant'].Radius[int(sol)]
            cost_ant = cost_ant + self.df['ant'].Cost[int(sol)]
            # Calculate Quality of Service
            # Find which segments are covered
            frac = sol - int(sol)
            to_x, to_y = self.geo_to[seg,:]
            from_x, from_y = self.geo_from[seg,:]
            idx = self.__theano_function(subsegs[0], subsegs[1], to_x, to_y, from_x, from_y, frac, radius)[0] == 1            
            covered_segs = covered_segs | idx
        # Combine costs
        cost_qos = self.total_traffic - self.df['subsegs'].Traffic.values[covered_segs].sum() 
        cost = cost_ant + cost_qos
        return cost


##################################################################################################



    
    


