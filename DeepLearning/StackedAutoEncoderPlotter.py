

from StackedAutoEncoder import StackedAutoEncoder


import matplotlib.pyplot as plt


class StackedAutoEncoderPlotter(StackedAutoEncoder):
    
    def __init__(self, obj):
        self.hiddens = obj.hiddens
        self.optimizers = obj.optimizers
        self.nepochs = obj.nepochs
        self.batch_size = obj.batch_size
        self.ninit = obj.ninit
        self.__scaler = obj.__StackedAutoEncoder_scaler
        self.__model = obj.__StackedAutoEncoder_model
        self.__encoder = obj.__StackedAutoEncoder_encoder
        self.trn_info = obj.trn_info
        self.__internal_nets = obj.__StackedAutoEncoder_internal_nets
        self.verbose = obj.verbose
        self.label = obj.label

    def plot_training_curves(self):
        for ilayer in self.trn_info.keys():
            plt.figure()
            metric = 'loss'
            plt.plot(self.trn_info[ilayer].epoch, self.trn_info[ilayer].history[metric], '-b', lw = 3, label='train')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.title('Net %i:%i:%i - error (training) = %.3e'%(self.__model.layers[ilayer-1].get_input_shape_at(0)[1],
                                                                self.__model.layers[ilayer-1].get_output_shape_at(0)[1],
                                                                self.__model.layers[ilayer-1].get_input_shape_at(0)[1],
                                                                self.trn_info[ilayer].history[metric][-1])) 
