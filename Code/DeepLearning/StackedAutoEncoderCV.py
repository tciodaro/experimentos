
import time
from sklearn import metrics
from sklearn import model_selection
from sklearn.externals import joblib

import numpy as np

from Sonar import StackedAutoEncoder as SAE


class StackedAutoEncoderCV(object):
    
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
    def fit(self, data, target=None, nclasses = 1):
        t0 = time.time()
        # Test x Development
        if target is None:
            target = np.ones(data.shape[0])
        kfold = None
        if nclasses == 1:
            kfold = model_selection.KFold(n_splits=4, random_state = self.random_seed)
        else:
            kfold = model_selection.StratifiedKFold(n_splits=4, random_state = self.random_seed)

        clf = SAE.StackedAutoEncoder(verbose=False)        

        grid = model_selection.GridSearchCV(clf, param_grid=self.grid_params, cv=kfold,
                                            n_jobs=self.njobs,
                                            scoring = {'mse': SAE.StackedAutoEncoderMSE,
                                                       'kl_div': SAE.StackedAutoEncoderScorer},
                                            refit = 'kl_div')
        grid.fit(data, target)
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
        self.best_params = dict([(k.replace('network__',''), v) for k,v in grid.best_params_.items()])
        self.results = grid.cv_results_
        self.best_index = grid.best_index_
        self.network = SAE.StackedAutoEncoder(**self.best_params)
        self.network.fit(data[self.cv_indexes[icv]['itrn']], target[self.cv_indexes[icv]['itrn']])
        self.mean_score = grid.cv_results_['mean_test_kl_div'][grid.best_index_]
        self.std_score = grid.cv_results_['std_test_kl_div'][grid.best_index_]
        print 'Total time: ', time.time()-t0
        print 'Result: %.3f +- %.3f'%(self.mean_score,self.std_score)
        

    def save(self, fname):
        print 'Saving CV to ', fname
        net_file = ''
        if self.network is not None:
            self.network.label = '-'.join([str(x) for x in self.network.hiddens])
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
        self.network = SAE.StackedAutoEncoder(**self.best_params)
        self.network.load(fname)
        
    def encode(self, data):
        return self.network.encode(data)
    
    def predict(self, data):
        return self.network.predict(data)
    
    def score(self, data, target = None):
        return self.network.score(data)



# end of file


