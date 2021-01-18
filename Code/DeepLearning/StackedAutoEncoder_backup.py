from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import backend

import matplotlib.pyplot as plt

import numpy as np
import time
from sklearn import metrics


class StackedAutoEncoder(object):
    
    def __init__(self, conf = None, verbose=False):
        self.__config = {}
        if conf is None:
            self.__config['hiddens']    = []
            self.__config['optimizers']    = []
            self.__config['nepochs']    = 500
            self.__config['batch_size'] = 100
            self.__config['patience']   = 100
            self.__config['ninit']      = 10
            self.__config['itrn']       = None
            self.__config['ival']       = None
        elif isinstance(conf, dict):
            for k,v in conf.items():
                self.__config[k] = v
        else:
            raise Exception('StackedAutoEncoder: conf parameter is not a dict')
            
        self.__model = None
        self.__encoder = None
        self.__trn_info = None
        self.__internal_nets = None
        self.training_error = -1
        self.validation_error = -1
        self.verbose = verbose
        
    """
        Set train indexes
    """
    def set_itrn(self, itrn):
        self.__config['itrn'] = itrn
    def set_ival(self, ival):
        self.__config['ival'] = ival
    
    """
        Set hidden layers and optimizers
    """
    def set_layers(self, layers):
        self.__config['hiddens'] = layers
    def set_optimizers(self, opts):
        self.__config['optimizers'] = opts
    
    
    """
        Fit the auto encoder to the data given
    """
    def fit(self, data):
        t0 = time.time()
        if 'hiddens' not in self.__config.keys() or self.__config['hiddens'] is None:
            raise Exception('StackedAutoEncoder: hidden layers parameter not set')
        
        # Over training layers
        npairs = int((len(self.__config['hiddens'])-1)/2.0)
        self.__trn_info = {}
        nnet   = {}
        itrn = self.__config['itrn']
        ival = self.__config['ival']
        if self.verbose:
            print 'Training %i layer pairs'%npairs
        X = data
        for ilayer in range(1, npairs+1):
            if self.verbose:
                print "\tLayer pair %i"%(ilayer)
                print "\t\tStructure = %i:%i:%i"%(self.__config['hiddens'][ilayer-1],
                                                  self.__config['hiddens'][ilayer],
                                                  self.__config['hiddens'][len(self.__config['hiddens']) - ilayer])
                print "\t\tActivations = tanh:%s"%('linear' if ilayer == 1 else 'tanh') # only first iteration's output is linear
            ###### Training
            # Different Initializations
            self.__trn_info[ilayer] = None
            best_perf = 1e9
            nnet[ilayer] = None
            for iinit in range(self.__config['ninit']):
                model = models.Sequential()
                # Create network structure
                model.add(layers.Dense(self.__config['hiddens'][ilayer],
                                       activation = 'tanh',
                                       input_shape = [self.__config['hiddens'][ilayer-1]],
                                       kernel_initializer = 'uniform'))
                model.add(layers.Dense(self.__config['hiddens'][len(self.__config['hiddens']) - ilayer],
                                       activation = 'linear' if ilayer == 1 else 'tanh',
                                       input_shape = [self.__config['hiddens'][ilayer]],
                                       kernel_initializer = 'uniform'))  
                # Training
                opt = None
                if self.__config['optimizers'][ilayer-1] == 'adam':
                    opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                elif self.__config['optimizers'][ilayer-1] == 'sgd':
                    opt = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                else:
                    raise Exception('StackedAutoEncoder: unknown optimizer for pair %i: %s'%(ilayer-1,
                                                                                             self.__config['optimizers'][ilayer-1]))
                model.compile(loss='mean_squared_error', optimizer=opt)
                earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=self.__config['patience'], verbose=0, mode='auto')
                # Should be done for each initialization
                init_trn_desc = model.fit(X[itrn], X[itrn], 
                                          epochs = self.__config['nepochs'], 
                                          batch_size = self.__config['batch_size'],
                                          callbacks = [earlyStopping], 
                                          verbose = self.__config['verbose'],
                                          validation_data = (X[ival], X[ival]),
                                          shuffle=False)
                # Get best
                if init_trn_desc.history['loss'][-1] < best_perf:
                    self.__trn_info[ilayer] = init_trn_desc
                    best_perf = init_trn_desc.history['loss'][-1]
                    nnet[ilayer] = model
            if self.verbose:
                print '\t\tTraining Error: %.3e'%(self.__trn_info[ilayer].history['loss'][-1])
            # Update input as the output of the hidden layer
            hidden_layer = backend.function([nnet[ilayer].layers[0].input],[nnet[ilayer].layers[0].output])
            X = hidden_layer([X])[0]
        # Put together final model
        self.__model = models.Sequential()
        self.__encoder = models.Sequential()
        for ilayer in range(1, npairs+1):
            # Encoder part
            self.__model.add(nnet[ilayer].layers[0])
            self.__encoder.add(nnet[ilayer].layers[0])
        for ilayer in range(npairs, 0, -1):
            # Decoder part
            self.__model.add(nnet[ilayer].layers[1])
        self.__internal_nets = nnet
        Y = self.__model.predict(data)
        self.training_error = metrics.mean_squared_error(data[itrn], Y[itrn])
        self.validation_error = metrics.mean_squared_error(data[ival], Y[ival])
        if self.verbose:
            print 'Final Reconstruction Error (training)  : %.3e'%(self.training_error)
            print 'Final Reconstruction Error (validation): %.3e'%(self.validation_error)
            print 'Training took %i s'%(time.time() - t0)

        
    def encode(self, data):
        return self.__encoder.predict(data)
    
    def predict(self, data):
        return self.__model.predict(data)
       
    def get_encoder(self):
        return self.__encoder
    
    def get_auto_encoder(self):
        return self.__model
    
    def plot_training_curves(self):
        for ilayer in self.__trn_info.keys():
            plt.figure()
            metric = 'loss'
            plt.plot(self.__trn_info[ilayer].epoch, self.__trn_info[ilayer].history[metric], '-b', lw = 3, label='train')
            plt.plot(self.__trn_info[ilayer].epoch, self.__trn_info[ilayer].history['val_'+metric], '--g', lw = 3, label='valid.')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.title('Net %i:%i:%i - error (training) = %.3e'%(self.__model.layers[ilayer-1].get_input_shape_at(0)[1],
                                                                self.__model.layers[ilayer-1].get_output_shape_at(0)[1],
                                                                self.__model.layers[ilayer-1].get_input_shape_at(0)[1],
                                                                self.__trn_info[ilayer].history[metric][-1]))



# end of file


