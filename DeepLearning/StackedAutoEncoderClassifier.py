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
from sklearn.externals import joblib
from scipy.stats import mstats

from Sonar import StackedAutoEncoderCV as SAE


def SPScorer(clf, data, target):
    Y = clf.predict(data)
    if np.unique(target).shape[0] == 2:
        Y = Y > 0
    else:    
        Y = np.argmax(Y, axis=1)
    effs = np.zeros(np.unique(target).shape[0])
    for cls in range(np.unique(target).shape[0]):
        effs[cls] = float((Y[target==cls]==cls).sum()) / (target==cls).sum()
    
    return mstats.gmean([mstats.gmean(effs), np.mean(effs)])


def PrecisionScorer(clf, data, target):
    Y = clf.predict(data)
    if np.unique(target).shape[0] == 2:
        Y = Y > 0
    else:    
        Y = np.argmax(Y, axis=1)
    return metrics.precision_score(target, Y, average='weighted')

def RecallScorer(clf, data, target):
    Y = clf.predict(data)
    if np.unique(target).shape[0] == 2:
        Y = Y > 0
    else:    
        Y = np.argmax(Y, axis=1)
    return metrics.recall_score(target, Y, average='weighted')


class StackedAutoEncoderClassifier(object):
    
    def __init__(self, hidden = [], optimizer = [], nepochs = 500, batch_size=100, ninit = 10, verbose=False, label = ''):
        self.hidden = hidden
        self.optimizer = optimizer
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.ninit=ninit 
        self.classifier = None
        self.trn_info = None
        self.verbose = verbose
        self.label = label
        


    """
        Fit the classifier
    """
    def fit(self, data, target, encoders):
        t0 = time.time()
        if self.hidden is None or self.hidden == 0:
            raise Exception('StackedAutoEncoderClassifier: hidden layers parameter not set')
        best_perf = 1e9
        # One hot encoding
        if np.unique(target).shape[0] == 2:
            one_hot_target = np.ones([target.shape[0],1])
            one_hot_target[target == np.unique(target)[0], 0] = -1
        else:
            one_hot_target = -np.ones([target.shape[0], np.unique(target).shape[0]])
            for i, cls in enumerate(np.unique(target)):
                one_hot_target[target == cls, i] = 1
        target = one_hot_target                
        # Different initializations
        for iinit in range(self.ninit):
            common_input = layers.Input(shape = [data.shape[1]])
            graphs = []
            for cls in encoders.keys():
                gr = common_input
                for lay in encoders[cls].layers:
                    lay.name = 'class_'+ cls + lay.name
                    lay.trainable = False
                    gr = lay(gr)
                graphs.append(gr)
            merge_layer = layers.concatenate(graphs, axis=1)
            hidden_layer = layers.Dense(self.hidden, activation='tanh')(merge_layer)
            out_layer = layers.Dense(target.shape[1], activation='tanh')(hidden_layer)
            new_model = models.Model(inputs=[common_input], outputs=out_layer)
            # Training
            opt = None
            if self.optimizer == 'adam':
                opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            elif self.optimizer == 'sgd':
                opt = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
            else:
                raise Exception('StackedAutoEncoderClassifier: unknown optimizer: %s'%(self.optimizer))
            new_model.compile(loss='mean_squared_error', optimizer=opt)
            init_trn_desc = new_model.fit(data, target, 
                                          epochs = self.nepochs, 
                                          batch_size = self.batch_size,
                                          verbose = False,
                                          shuffle = False)
            if init_trn_desc.history['loss'][-1] < best_perf:
                    self.trn_info = dict(init_trn_desc.history)
                    self.trn_info['epochs'] = init_trn_desc.epoch
                    best_perf = init_trn_desc.history['loss'][-1]
                    self.classifier = new_model
        if self.verbose:
            print 'Final Classification Accuracy (training)  : %.3e'%(best_perf)
            print 'Training took %i s'%(time.time() - t0)
    
        
    def save(self, fname):
        fname =  fname + '_' + self.label
        obj = {
            'hiddens': self.hidden,
            'optimizer': self.optimizer,
            'nepochs': self.nepochs,
            'batch_size': self.batch_size,
            'ninit': self.ninit,
            'trn_info': self.trn_info,
            'label': self.label
        }   
        joblib.dump(obj, fname+'_info.jbl')
        self.classifier.save(fname+'_model.ker')
        return fname
        
        
    def load(self, fname):
        objs = joblib.load(fname + '_info.jbl')
        for parameter, value in objs.items():
            setattr(self, parameter, value)
        # Load Keras model
        self.classifier = models.load_model(fname+'_model.ker')

    def predict(self, data):
        return self.classifier.predict(data)
    
    def score(self, data, target = None):
        return SPScorer(self.classifier, data, target)
    
    def get_params(self, deep=True):        
        return {'hidden': self.hidden,
                'optimizer': self.optimizer,
                'nepochs': self.nepochs,
                'batch_size': self.batch_size,
                'ninit': self.ninit}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
        
        
        
        
        
        
        
        
        
# END OF FILE












    