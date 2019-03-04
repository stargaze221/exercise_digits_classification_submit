'''
main module
run the following commands in terminal:
(1) Test the model saved in 'hyper_param.json' and 'Saved_CNN_Mdl.pt'
    $ python main.py test

(2) Train the model with the hyper-parameters in 'hyper_param.json'.
    $ python main.py optimize classifier

(3) Optimize the hyper-parameter and save in 'hyper_param.json'.
    $ python main.py optimize hyperparam
'''

import sys
import json
import torch
from train import Trainer



def optimize_hyperparam():
    # 0. Initialize a trainer.
    trainer = Trainer('log')
    # 1. Run hyper-parameters optimization.
    best = trainer.optimize_hyperparam()
    # 2. Save the hyper-parameters.
    with open('best_hyper_param.json', 'w', encoding="utf8") as outfile:
        json.dump(best, outfile)
    print('Hyper-parameter optimization is finished.')


def train_with_the_hyperparam():
    # 0. Initialize a trainer.
    trainer = Trainer('log')    
    # 1. Load the hyper-parameters saved in a JSON file.
    with open('hyper_param.json', 'r', encoding="utf8") as readfile:
        hyperparam = json.load(readfile)
    _, _, CNN = trainer.train(hyperparam['l2_weight'], hyperparam['n_layer'], hyperparam['n_epoch'])
    # 2. Save the model.
    torch.save(CNN.state_dict(), 'Saved_CNN_Mdl.pt')

    
def train():
    # 0. Initialize a trainer.
    trainer = Trainer('log')
    # 1. Train with the default hyper-parameters.
    _, _, CNN = trainer.train()
    # 2. Save the model.
    torch.save(CNN.state_dict(), 'Saved_CNN_Mdl.pt')


def test():
    # 0. Initialize a trainer.
    trainer = Trainer('log')
    # 1. Test the result.
    f_param='Saved_CNN_Mdl.pt'
    f_hyperparam='hyper_param.json'
    trainer.test(f_param, f_hyperparam)



if __name__ == "__main__":

    if len(sys.argv) == 1:
        train()
    else:
        arg1 = sys.argv[1]

        if arg1 == 'optimize':
            arg2 = sys.argv[2]
            if arg2 == 'hyperparam':
                optimize_hyperparam()

            elif arg2 == 'classifier':
                train_with_the_hyperparam()
            else:
                print('The command is not found. Try again with the other commands.')

        elif arg1 == 'test':
            test()
        else:
            print('The command is not found. Try again with the other commands.')