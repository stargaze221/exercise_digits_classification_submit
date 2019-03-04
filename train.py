'''
This module contains the definition of the class, "Trainer". The class has methods for training and validation by loading data, running the optimization, auto-tuning of hyper-parameters, and validation.
'''

import torch.nn as nn
import torch
from model import ConvNetwork
from util import Logger, load_mnist_data
import json


NB_DIGITS = 10 # Number of labels, i.e. 0, 1, 2, ... , 9.
N_PIXELS = 28 # The hand-writing images are wth resolutions at (28 x 28).

BATCHSIZE = 10 # Size of the minibatch for stochastic gradient descent (SGD) algorithm. 
LEARNING_RATE = 0.001 # Step size for updating the paraemters in SGD for the Adam optimizer.

class Trainer:
    '''
    Trainer class has the methods for training and validation.
    '''
    def __init__(self, f_name_log):
        '''
        Initialization is consist of:
            0. Check the device.
            1. Set up a logger.
            2. Load data.
        '''
        # 0. Check the device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('The device, ', self.device, ', is used.')

        # 1. Set up a logger.
        header = ['l2_reg_wt', 'n_layers', 'n_epoch', 'valid_accuracy', 'isnan']
        self.logger = Logger(header, f_name_log)

        # 2. Load data.
        X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, X_TEST, Y_TEST = load_mnist_data()
        self.data = {'x_train':X_TRAIN/255., 'y_train':Y_TRAIN, 'x_valid':X_VALID/255., 'y_valid':Y_VALID, 'x_test':X_TEST/255., 'y_test':Y_TEST} #Normalize the image by dividing 255.


    def get_cross_entropy_loss(self, CNN, X, Y):
        '''
        Argument:
            CNN is the comvolutional neural network (CNN) object implemented using Pytorch.
            X and Y are Pytorch Tensor.
        Return:
            Cross-Entroy-Loss is Pytorch Tensor which contains the auto-differentiable computation graph.
        '''
        y_onehot = (torch.Tensor(len(X), NB_DIGITS).to(self.device)).zero_()
        y_onehot.scatter_(1, Y.view(-1, 1), 1)
        y_pred_dist = CNN.forward(X.view(-1, 1, N_PIXELS, N_PIXELS))

        # 0. Cross-entropy calculation.
        cross_entropy = -torch.sum(y_onehot*torch.log(y_pred_dist))/len(X) 

        return cross_entropy


    def get_classification_err(self, CNN, X, Y):
        '''
        Argument:
            CNN is the comvolutional neural network (CNN) object implemented using Pytorch.
            X and Y are Pytorch Tensor.
        Return:
            Cross-Entroy-Loss is Pytorch Tensor which contains the auto-differentiable computation graph.
        '''
        # 0. One hot tensor determined by the classification labels, Y.
        y_onehot = (torch.Tensor(len(X), NB_DIGITS).to(self.device)).zero_()
        y_onehot.scatter_(1, Y.view(-1, 1), 1)

        # 1. One hot tensor determined by the CNN prediction given the images, X.
        y_pred_label = torch.argmax(CNN.forward(X.view(-1, 1, N_PIXELS, N_PIXELS)), dim=1)
        y_onehot_pred = (torch.Tensor(len(X), NB_DIGITS).to(self.device)).zero_()
        y_onehot_pred.scatter_(1, y_pred_label.view(-1, 1), 1)

        # 2. Correctness ratio
        correct_ratio = float(torch.sum(y_onehot*y_onehot_pred)/len(X))
        
        return 1. - correct_ratio # Error ratio.

    
    def train(self, WEIGHT_DECAY=0.00001, N_LAYERS=3, N_EPOCH=10):
        '''
        Argument:
            WEIGHT_DECAY is l2 regularization.
            N_LAYERS is number of CNN layers.
            N_EPOCH is number of epochs (A turn of using the train dataset in SGD counts for an epoch).

        Return:
            VALID_LOSS is cross entropy loss. 
            VALID_CLASSFICIATION_ERROR is frequency of errors.
            CNN is the pytorch obejct.  
        '''
        # 0. Initialize the convolutional neural network
        CNN = ConvNetwork(n_layers=N_LAYERS).to(self.device)

        # 1. Initialize the Adam stochastic gradient descent (SGD) optimizer.
        OPTIMIZER = torch.optim.Adam(CNN.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)

        # 2. Load train data as Pytorch Tensors into the device. 
        X_TRAIN = torch.Tensor(self.data['x_train']).to(self.device)
        Y_TRAIN = torch.LongTensor(self.data['y_train']).to(self.device)

        # 3. Iterate SGD with epochs and minibatches.
        ITERATION = 0
        ISNAN = False

        for epoch in range(N_EPOCH):
            running_loss = 0.0
            for _x, _y in zip(X_TRAIN.split(BATCHSIZE), Y_TRAIN.split(BATCHSIZE)):
                ITERATION = ITERATION + 1
                OPTIMIZER.zero_grad() # Reset the gradient to zero values.
                loss = self.get_cross_entropy_loss(CNN, _x, _y) # Calculate cross-entropy loss.
                
                if torch.isnan(loss): # Stop the iterations if NAN happens.
                    ISNAN = True
                    break

                loss.backward() # Back propogate to calculate the gradient for the loss respect to the parameters of CNN.
                OPTIMIZER.step() # Update the parameters with the gradients using the Adam optmizer.

                # 3-0. Print the loss every 100 iterations.
                running_loss = running_loss + float(loss.data)
                if ITERATION % 50 == 49:
                    print('[%d, %5d] loss: %.3f' %(epoch+1, ITERATION+1, running_loss/200))
                    running_loss = 0.0
        print('Finshed training.')

        # 4. Validate the model parameters with the validation data.
        X_VALID = torch.Tensor(self.data['x_valid']).to(self.device)
        Y_VALID = torch.LongTensor(self.data['y_valid']).to(self.device)
        
        VALID_LOSS = float(self.get_cross_entropy_loss(CNN, X_VALID, Y_VALID))
        VALID_CLASSFICIATION_ERROR = float(self.get_classification_err(CNN, X_VALID, Y_VALID))

        # 5. Log the hyper-parameters and classification correctness with the valication data.
        self.logger.write_a_row([WEIGHT_DECAY, N_LAYERS, N_EPOCH, 1 - VALID_CLASSFICIATION_ERROR, ISNAN])

        return VALID_LOSS, VALID_CLASSFICIATION_ERROR, CNN


    def optimize_hyperparam(self):
        '''
        Return:
            Hyper-prameters determined to minimize the classification error with validation data by a sample based optimization.
        
        The tool, "hyperopt," is a Bayesian optimization tool which recursively updates the initial guess of the shape of the cost function as it accumulates trials.
        '''

        # 0. Import Hyperopt.
        from hyperopt import fmin, tpe, hp, Trials

        # 1. Define the search space.
        space = {
                    'log10_l2': hp.uniform('log10_l2', -5, -2),
                    'n_layers-2': hp.randint('n_layers-2', 4),
                    'n_epoch-5': hp.randint('n_epoch-5', 10)
                }

        # 2. Define the cost function.
        def f(space):    
            l2_weight = 10**space['log10_l2']
            n_layers = space['n_layers-2']+2
            n_epoch = space['n_epoch-5']+5
            loss, error_ratio, _ = self.train(l2_weight, n_layers, n_epoch)
            return error_ratio*100.

        # 3. Tools for monitoring the optimization with Hyperopt.
        trials = Trials()

        # 4. Run the hyper-parameter optimization.
        best = fmin(
                    fn=f,  # "Loss" function to minimize
                    space=space,  # Hyperparameter space
                    algo=tpe.suggest,  # Tree-structured Parzen Estimator (TPE)
                    trials=trials,
                    max_evals=100  # Perform 1000 trials
                    )
        print("Found minimum after 200 trials:")

        # 5. Save the determined hyper-parameters to return.    
        best = {
                    'l2_weight': 10**best['log10_l2'],
                    'n_layer': best['n_layers-2']+2,
                    'n_epoch': best['n_epoch-5']+5
                 }
        print(best)

        return best


    def test(self, f_param='Saved_CNN_Mdl.pt', f_hyperparam='hyper_param.json'):
        # 0. Load hyper-parameters.
        with open(f_hyperparam, 'r', encoding="utf8") as readfile:
            hyperparam = json.load(readfile)
        N_LAYERS = hyperparam['n_layer']

        # 1. Initialize the convolutional neural network
        CNN = ConvNetwork(n_layers=N_LAYERS).to(self.device)

        # 2. Load the saved params.
        CNN.load_state_dict(torch.load(f_param, map_location=lambda storage, loc: storage))
        print('Model loaded succesfully.')

        # 3. Show the errors with validation and test dataset
        print('Classification errors:')

        # 3-0. Validation dataset
        X_VALID = torch.Tensor(self.data['x_valid']).to(self.device)
        Y_VALID = torch.LongTensor(self.data['y_valid']).to(self.device)
        VALID_CLASSFICIATION_ERROR = float(self.get_classification_err(CNN, X_VALID, Y_VALID))
        print('with Valid. Data: ', VALID_CLASSFICIATION_ERROR)

        # 3-1. Test dataset 
        X_TEST = torch.Tensor(self.data['x_test']).to(self.device)
        Y_TEST = torch.LongTensor(self.data['y_test']).to(self.device)
        TEST_CLASSFICIATION_ERROR = float(self.get_classification_err(CNN, X_TEST, Y_TEST))
        print('with Test. Data: ', TEST_CLASSFICIATION_ERROR)

        