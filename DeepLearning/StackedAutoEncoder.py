import os

os.environ['KERAS_BACKEND'] = 'theano'

from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import backend
from scipy import stats

import numpy as np
import time
from sklearn import metrics
from sklearn import preprocessing
import joblib
import uuid


def StackedAutoEncoderScorer(clf, data, target):
    return -clf.score(data) #Negative: gridsearch will try to make it max

def StackedAutoEncoderMSE(clf, data, target):
    Y = clf.predict(data)
    return metrics.mean_squared_error(data, Y)

class StackedAutoEncoder(object):
    
    def __init__(self, hiddens = [], optimizers = [], nepochs = 500, batch_size=100, ninit = 10, verbose=False, label = ''):
        self.hiddens = hiddens
        self.optimizers = optimizers
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.ninit=ninit 
        self.__model = None
        self.__encoder = None
        self.__decoder = None
        self.trn_info = None
        self.__internal_nets = None
        self.verbose = verbose
        self.label = label
    
    
    """
        Fit the auto encoder to the data given
    """
    def fit(self, data, target=None, validation_data = None):
        t0 = time.time()
        if self.hiddens is None or len(self.hiddens) == 0:
            raise Exception('StackedAutoEncoder: hidden layers parameter not set')
        # Over training layers
        npairs = int((len(self.hiddens)-1)/2.0)
        self.trn_info = {}
        nnet   = {}
        if self.verbose:
            print('Training %i layer pairs'%npairs)
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(data)
        V = None
        if validation_data is not None:
            V = scaler.transform(validation_data)
        # Per layer
        for ilayer in range(1, npairs+1):
            if self.verbose:
                print("\tLayer pair %i"%(ilayer))
                print("\t\tStructure = %i:%i:%i"%(self.hiddens[ilayer-1],
                                                  self.hiddens[ilayer],
                                                  self.hiddens[len(self.hiddens) - ilayer]))
                print("\t\tActivations = tanh:%s"%('linear' if ilayer == 1 else 'tanh')) # only first iteration's output is linear
            ###### Training
            # Different Initializations
            self.trn_info[ilayer] = None
            best_perf = 1e9
            nnet[ilayer] = None
            for iinit in range(self.ninit):
                model = models.Sequential()
                # Create network structure
                model.add(layers.Dense(self.hiddens[ilayer],
                                       activation = 'tanh',
                                       input_shape = [self.hiddens[ilayer-1]],
                                       kernel_initializer = 'uniform'))
                model.add(layers.Dense(self.hiddens[len(self.hiddens) - ilayer],
                                       activation = 'linear' if ilayer == 1 else 'tanh',
                                       input_shape = [self.hiddens[ilayer]],
                                       kernel_initializer = 'uniform'))  
                # Training
                opt = None
                if self.optimizers[ilayer-1] == 'adam':
                    opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                elif self.optimizers[ilayer-1] == 'sgd':
                    opt = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                else:
                    raise Exception('StackedAutoEncoder: unknown optimizer for pair %i: %s'%(ilayer-1,
                                                                                             self.optimizers[ilayer-1]))
                model.compile(loss='mean_squared_error', optimizer=opt)
                # Should be done for each initialization
                init_trn_desc = model.fit(X, X,
                                          validation_data = (V,V) if V is not None else None,
                                          epochs = self.nepochs, 
                                          batch_size = self.batch_size,
                                          verbose = self.verbose,
                                          shuffle=False)
                # Get best
                if init_trn_desc.history['loss'][-1] < best_perf:
                    self.trn_info[ilayer] = dict(init_trn_desc.history)
                    best_perf = init_trn_desc.history['loss'][-1]
                    nnet[ilayer] = model
            if self.verbose:
                print('\t\tTraining Error: %.3e'%(self.trn_info[ilayer]['loss'][-1]))
            # Update input as the output of the hidden layer
            hidden_layer = backend.function([nnet[ilayer].layers[0].input],[nnet[ilayer].layers[0].output])
            X = hidden_layer([X])[0]
            if validation_data is not None:
                V = hidden_layer([V])[0]
        # Absorb scaling
        s = scaler.scale_
        m = scaler.mean_
        # Input
        W = nnet[1].layers[0].get_weights()[0]
        B = nnet[1].layers[0].get_weights()[1]
        nnet[1].layers[0].set_weights( (W / s.reshape((s.shape[0],1)), B - np.dot(W.T,(m/s))))
        # Output
        W = nnet[1].layers[-1].get_weights()[0]
        B = nnet[1].layers[-1].get_weights()[1]
        nnet[1].layers[-1].set_weights((W * s, B*s + m))
        # Put together final model
        self.__model = models.Sequential()
        self.__encoder = models.Sequential()
        self.__decoder = models.Sequential()
        for ilayer in range(1, npairs+1):
            # Encoder part
            self.__model.add(nnet[ilayer].layers[0])
            self.__encoder.add(nnet[ilayer].layers[0])
        for ilayer in range(npairs, 0, -1):
            # Decoder part
            self.__model.add(nnet[ilayer].layers[1])
            self.__decoder.add(nnet[ilayer].layers[1])
        self.__internal_nets = nnet
        score = self.score(data)
        if self.verbose:
            print('Final Reconstruction Error (training)  : %.3e'%(score))
            print('Training took %i s'%(time.time() - t0))

    def save(self, fname):
        #fname =  fname + '_' + str(uuid.uuid4()) + self.label
        fname =  fname + '_' + self.label
        obj = {
            'hiddens': self.hiddens,
            'optimizers': self.optimizers,
            'nepochs': self.nepochs,
            'batch_size': self.batch_size,
            'ninit': self.ninit,
            'trn_info': self.trn_info,
            'label': self.label
        }   
        joblib.dump(obj, fname+'_info.jbl')
        self.__model.save(fname+'_model.ker')
        return fname
        
        
    def load(self, fname):
        objs = joblib.load(fname + '_info.jbl')
        for parameter, value in objs.items():
            setattr(self, parameter, value)
        # Load Keras model
        self.__model = models.load_model(fname+'_model.ker')
        self.__encoder = models.Sequential()
        npairs = int((len(self.hiddens)-1)/2.0)
        self.__internal_nets = {}
        for ilayer in range(1, npairs+1):
            # individual trained nets
            self.__internal_nets[ilayer] = models.Sequential()
            self.__internal_nets[ilayer].add(self.__model.layers[ilayer-1])
            self.__internal_nets[ilayer].add(self.__model.layers[-ilayer])
            # Encoder
            self.__encoder.add(self.__model.layers[ilayer-1])
        


    def encode(self, data):
        return self.__encoder.predict(data)

    def decode(self, data):
        return self.__decoder.predict(data)
    
    def predict(self, data):
        Y = self.__model.predict(data)
        return Y
    
    def score(self, data, target = None):
        return self.calculate_score(self.predict(data), data)

    def calculate_score(self, Y, X):
        X = (X.T / np.sum(X, 1)).T
        Y = (Y.T / np.sum(Y, 1)).T
        X = (X.T - np.min(X, 1)).T
        Y = (Y.T - np.min(Y, 1)).T
        return np.mean(stats.entropy(X.T + 1e-9,Y.T + 1e-9)) # KL Divergence
    
    def get_params(self, deep=True):        
        return {'hiddens': self.hiddens,
                'optimizers': self.optimizers,
                'nepochs': self.nepochs,
                'batch_size': self.batch_size,
                'ninit': self.ninit}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def get_encoder(self):
        return self.__encoder
    
    def get_decoder(self):
        return self.__decoder

    def get_auto_encoder(self):
        return self.__model
    
    



# end of file


