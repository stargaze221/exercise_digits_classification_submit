'''
Utility funcitons and classes:
1. The function, "Load_mnist_data", saves the data from LeCun's website and returns the data cased as Numpy arrays.
2. The class, "Logger", has a method to save the data into a csv file.
'''
import csv
import os



def load_mnist_data():
    '''
    This funciton load the MNIST data.
    We divide the training data into training and validation with the ratio 4:1.
    '''
    import os
    import wget
    import gzip
    import idx2numpy
    import numpy

    if not os.path.exists('data'):
        os.makedirs('data')
        print('Downloading MNIST data files')
        wget.download('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', out='data')
        wget.download('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', out='data')
        wget.download('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', out='data')
        wget.download('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', out='data')

    ### Load the train data ###
    _f = gzip.open('data/train-images-idx3-ubyte.gz')
    x_train = idx2numpy.convert_from_file(_f)
    _f = gzip.open('data/train-labels-idx1-ubyte.gz')
    y_train = idx2numpy.convert_from_file(_f)

    _f = gzip.open('data/t10k-images-idx3-ubyte.gz')
    x_test = idx2numpy.convert_from_file(_f)
    _f = gzip.open('data/t10k-labels-idx1-ubyte.gz')
    y_test = idx2numpy.convert_from_file(_f)

    ### Get validation set ###
    random_indices = numpy.arange(len(y_train))
    numpy.random.shuffle(random_indices)
    x_train = x_train[random_indices]
    y_train = y_train[random_indices]

    x_valid = x_train[:len(y_train)//4]
    y_valid = y_train[:len(y_train)//4]

    x_train = x_train[len(y_train)//4:]
    y_train = y_train[len(y_train)//4:]

    return x_train, y_train, x_valid, y_valid, x_test, y_test



class Logger:
    '''
    It initializes a CSV file to save the progress of training and defines a method to write a row of logging variables in the file.
    '''

    def __init__(self, header, f_name):
        self.header = header
        self.f_path = './log/' + f_name + '.csv'

        if not os.path.exists('log'):
            os.makedirs('log')

        with open(self.f_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(header)

    def write_a_row(self, a_row):
        with open(self.f_path, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(a_row)