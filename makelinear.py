'''This module has some simple functionality for making a fake data set based
on a linear function.
'''
import numpy as np

from datamaker import create_datasets

def _linear_function(indata):
    '''Simply the sum of the input data'''
    return sum(indata)
    
def make_random_indata(minimum, maximum, num, cols):
    '''minimum is the lowest number which should be possible
    maximum is added to minimum. the real maximum will be min + max.
    num = number of data points
    cols = number of columns
    '''
    return maximum * np.random.random_sample((num, cols)) + minimum
    
def make_linear_output(filename = None, num = 200, cols = 2):
    '''Makes a linear data set.'''
    indataset = make_random_indata(1, 20, num, cols)
    
    return create_datasets(_linear_function, indataset, filename)
    
if __name__ == '__main__':
    (perfect, perfect_noisy, censored, censored_noisy) = make_linear_output('linear_test_output')
    