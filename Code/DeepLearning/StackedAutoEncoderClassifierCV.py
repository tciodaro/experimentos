

import time
from sklearn import metrics
from sklearn import model_selection
from sklearn.externals import joblib

import numpy as np

from Sonar import StackedAutoEncoderClassifier as SAE


class StackedAutoEncoderClassifierCV(object):
    
    def __init__(self, grid_params = None, nfolds=5,njobs = 1, random_seed = None, verbose = False):
        self.grid_params = grid_params
        self.network = None
        self.results = {}
        self.best_params = []
        self.best_index = -1
        
        self.verbose = verbose
        self.nfolds = nfolds
        self.random_seed = random_seed
        self.cv_indexes = []
        self.njobs = njobs
        self.mean_score = 1e9
        self.std_score = 1e9

    
    """
        Fit the grid search 
    """
    def fit(self, data, target, encoders):
        t0 = time.time()
        kfold = model_selection.StratifiedKFold(n_splits=4, random_state = self.random_seed)
        clf = SAE.StackedAutoEncoderClassifier(verbose=False)
        grid = model_selection.GridSearchCV(clf, param_grid=self.grid_params, cv=kfold,
                                            n_jobs=self.njobs,
                                            scoring = {'sp': SAE.SPScorer,
                                                       'precision': SAE.PrecisionScorer,
                                                       'recall': SAE.RecallScorer},
                                            refit = 'sp')
        grid.fit(data, target, **{'encoders': encoders})
        # Find the best CV
        icv = -1
        best_score = -1e9
        for k,v in grid.cv_results_.items():
            if k.find('split') != -1 and k.find('_test_') != -1:
                if best_score < v[grid.best_index_]:
                    best_score = v[grid.best_index_]
                    icv = int(k[k.find('split')+5 : k.find('_')])
        # Get original indexes
        for i, (itrn, ival) in enumerate(kfold.split(data, target)):
            self.cv_indexes.append({'itrn': itrn, 'ival': ival})
        # Fix parameter names
        self.best_params = grid.best_params_
        self.results = dict(grid.cv_results_)
        self.best_index = grid.best_index_
        self.network = SAE.StackedAutoEncoderClassifier(**self.best_params)
        self.network.fit(data[self.cv_indexes[icv]['itrn']], target[self.cv_indexes[icv]['itrn']], **{'encoders': encoders})
        self.mean_score = grid.cv_results_['mean_test_sp'][grid.best_index_]
        self.std_score = grid.cv_results_['std_test_sp'][grid.best_index_]
        print 'Total time: ', time.time()-t0
        print 'Result: %.3f +- %.3f'%(self.mean_score,self.std_score)
        

    def save(self, fname):
        print 'Saving CV to ', fname
        net_file = ''
        if self.network is not None:
            self.network.label = 'h'+str(self.network.hidden)
            net_file = self.network.save(fname)
        objs = {}
        for k,v in self.__dict__.items():
            objs[k] = v
        # Keras model cannot be saved with joblib. Save the refname from StackedAutoEncoder
        # for later loading.
        objs['network'] = net_file
        joblib.dump(objs, fname, compress = 9)
        

    def load(self, fname):
        print 'Loading from ', fname
        obj = joblib.load(fname)
        for parameter, value in obj.items():
            setattr(self, parameter, value)
        # Load Keras
        fname = self.network
        self.network = SAE.StackedAutoEncoderClassifier(**self.best_params)
        self.network.load(fname)
        
    def predict(self, data):
        return self.network.predict(data)
    
    


# end of file







