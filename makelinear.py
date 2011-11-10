'''This module has some simple functionality for making a fake data set based
on a linear function.
'''
import numpy as np

from datamaker import create_datasets, create_datasets_with_input_noise

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
    
def make_linear_output(filename = None, num = 2000, cols = 4):
    '''Makes a linear data set.'''
    indataset = make_random_indata(1, 20, num, cols)
    
    return create_datasets(_linear_function, indataset, filename=filename)
    
def make_linear_output_with_input_noise(filename = None, num = 2000, cols = 4):
    '''Makes a linear data set but the noise is attributed to noise in the inputs, not in the outputs.'''
    indataset = make_random_indata(1, 20, num, cols)
    (a, b, c) = create_datasets_with_input_noise(_linear_function, indataset, filename=filename)
    return (a, b, indataset, c)
    
if __name__ == '__main__':
    #(perfect, perfect_noisy, censored, censored_noisy) = make_linear_output('linear_test_output.txt')
    (perfect, censored, indata, indata_noisy) = make_linear_output_with_input_noise('linear_test_output.txt')
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(perfect[:, 0], censored[:, 0], c = 'g', marker = 's')
    plt.plot(perfect[:, 0], perfect[:, 0], 'r-')
    plt.title('perfect vs censored')

    plt.figure()
    plt.scatter(indata[:, 0], indata_noisy[:, 0], c = 'g', marker = 's')
    plt.plot(indata[:, 0], indata[:, 0], 'r-')
    plt.title('in0 non vs censored')
    
    plt.figure()
    plt.scatter(indata[:, 1], indata_noisy[:, 1], c = 'g', marker = 's')
    plt.plot(indata[:, 1], indata[:, 1], 'r-')
    plt.title('in1 non vs censored')
    
    plt.figure()
    plt.scatter(indata[:, 2], indata_noisy[:, 2], c = 'g', marker = 's')
    plt.plot(indata[:, 2], indata[:, 2], 'r-')
    plt.title('in2 non vs censored')
    
    plt.figure()
    plt.scatter(indata[:, 3], indata_noisy[:, 3], c = 'g', marker = 's')
    plt.plot(indata[:, 3], indata[:, 3], 'r-')
    plt.title('in3 non vs censored')
    
    plt.show()
    