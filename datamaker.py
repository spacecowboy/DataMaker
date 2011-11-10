'''
This module is designed to create data sets for machine learning. Primarily
data intended to simulate censored survival data.
'''

from random import random, uniform, sample
import re
import numpy as np

__noise_level = 0.25

def _make_perfect(function, indata):
    '''y = f(x)
    Returns an array, [y, event] where event will be 1 since this data will be
    uncensored.
    '''
    return [function(indata), 1]

def _make_perfect_noisy(function, indata, minimum, maximum):
    '''y = f(x) +- noise
    Adds some noise to the otherwise perfect data.
    y is guaranteed to be greater than zero. Also, it is guaranteed to be
    greater than minimum.
    Returns an array, [y, event] where event will be 1 since this data will be
    uncensored.
    
    Minimum is the lowest value that the output is allowed to assume with noise
    Maximum is the largest value of y in the perfect data set.
    '''
    output = function(indata)
    if output - minimum < __noise_level * maximum:
        noise = uniform(minimum - output, __noise_level * maximum)
    else:
        noise = uniform(-__noise_level * maximum, __noise_level * maximum)
    return [output + noise, 1]

def _make_censored(function, indata, minimum):
    '''y = f(x) - rand
    Same as perfect data except that the output will be censored, meaning
    y will be less than f(x). Less by some random value.
    y is guaranteed to be greater than zero. Also, it is guaranteed to be
    greater than minimum.
    Returns an array, [y, event] where event is 0 because this data will be
    censored.
    
    Minimum is the lowest value that the output is allowed to assume with noise
    Smallest censoring will be 10% of the perfect output.
    '''
    output = function(indata)
    if output * 0.9 > minimum:
        censored = output - uniform(output * 0.1, output - minimum)
    else:
        censored = output - uniform(0, output - minimum)
    return [censored, 0]

def _make_censored_noisy(function, indata, minimum, maximum):
    '''y = f(x) - rand + noise
    Same as makeCensored but will then add some random noise as well. It adds
    the noise after censoring to make sure that the censoring doesn't get rid
    of the noise.
    y is guaranteed to be greater than zero. Also, it is guaranteed to be
    greater than minimum.
    Returns an array, [y, event] where event is 0 because this data is censored.
    
    Minimum is the lowest value that the output is allowed to assume with noise
    Smallest censoring will be 10% of the perfect output.
    Maximum is the largest value of y in the perfect data set.
    '''
    output = function(indata)
    #Make sure we don't censor past the minimum
    if output * 0.9 > minimum:
        censored = output - uniform(output * 0.1, output - minimum)
    else:
        censored = output - uniform(0, output - minimum)
    
    if censored - minimum < __noise_level * maximum:
        noise = uniform(minimum - censored, __noise_level * maximum)
    else:
        noise = uniform(-__noise_level * maximum, __noise_level * maximum)    
    
    return [censored + noise, 0]

def _savefile(filename, indataset, outdataset):
    '''Will save the data set in a file with the name and path specified.
    No headers will be entered but the structure will tab-separated like:
        X1, X2, ..., time, event
    '''
    with open(filename, 'w') as filehandle:
        for indata, outdata in zip(indataset, outdataset):
            for data in indata:
                filehandle.write(str(data) + "\t")
            filehandle.write(str(outdata[0]) + "\t")
            filehandle.write(str(outdata[1]) + "\n")

def create_datasets(function, indataset, censoredratio = 0.5,
                   filename = None):
    '''Given a function, and some in_data, this function will return a tuple, 
    (perfect, perfect_noisy, censored, censored_noisy) where each of those is
    a numpy array of shape (x, 2). first column is output, second is censoring
    variable (0 or 1).
    
    If given a filename, it will save the complete sets (including input) to
    files with the names: filename_perfect, filename_perfectnoisy,
    filename_censored, filename_censorednoisy (extension will be the same if
    one is specified in filename).
    
    censoredratio can be set to a value between 0 and 1 if a different amount
    of censoring is desired.
    
    Noise is set to 25% of the difference between lowest and highest output.
    
    Censored and noisy data is guaranteed to be greater than zero. Also, it is
    guaranteed to be greater than the smallest value in the perfect data set.
    '''
    perfect = np.empty((0, 2), dtype=float)
    perfect_noisy = np.empty((0, 2), dtype=float)
    censored = np.empty((0, 2), dtype=float)
    censored_noisy = np.empty((0, 2), dtype=float)
    
    #Need to two loops because I need the smallest value in perfect for
    #noise and censoring
    min_value = None
    max_value = None
    for indata in indataset:
        val = [_make_perfect(function, indata)]
        perfect = np.append(perfect, val, axis=0)
        if min_value is None or perfect[-1, 0] < min_value:
            min_value = perfect[-1, 0]
        if max_value is None or perfect[-1, 0] > max_value:
            max_value = perfect[-1, 0]
    
    for indata in indataset:
        perfect_noisy = np.append(perfect_noisy,
                                  [_make_perfect_noisy(function, indata, min_value, max_value)],
                                   axis=0)
        
        if random() < censoredratio:
            censored = np.append(censored,
                                 [_make_censored(function, indata, min_value)],
                                  axis=0)
            censored_noisy = np.append(censored_noisy,
                                       [_make_censored_noisy(function, indata, min_value, max_value)],
                                        axis=0)
        else:
            censored = np.append(censored, [_make_perfect(function, indata)],
                                            axis=0)
            censored_noisy = np.append(censored_noisy,
                                       [_make_perfect_noisy(function, indata, min_value, max_value)],
                                        axis=0)
    
    if filename is not None:
        m = re.search('(?<=.)\.[^\.]+$', filename)
        #Will include the .
        extension = m.group(0) if m is not None else ''
        #Remove extension from filename
        filename = re.sub('(?<=.)\.[^\.]+$', '', filename)
        #Save files
        _savefile(filename + '_perfect' + extension, indataset, perfect)
        _savefile(filename + '_perfectnoisy' + extension, indataset,
                  perfect_noisy)
                  
        _savefile(filename + '_censored' + extension, indataset, censored)
        _savefile(filename + '_censorednoisy' + extension, indataset,
                  censored_noisy)
    
    return (perfect, perfect_noisy, censored, censored_noisy)
    
def create_datasets_with_input_noise(function, indataset, censoredratio = 0.5, filename = None):
    '''Does the same as create_dataset but the noise is added to the input and not to the output.
    Returns a tuple with (perfect, censored, noisy_indataset)
    '''
    #Use create dataset
    (perfect, _, censored, _) = create_datasets(function, indataset, censoredratio)
    #Create noisy indata
    noisy_indataset = add_noise_to(indataset)
    
    if filename is not None:
        m = re.search('(?<=.)\.[^\.]+$', filename)
        #Will include the .
        extension = m.group(0) if m is not None else ''
        #Remove extension from filename
        filename = re.sub('(?<=.)\.[^\.]+$', '', filename)
        #Save perfect and censored with the indata we've got
        _savefile(filename + '_perfect' + extension, indataset, perfect)
        _savefile(filename + '_censored' + extension, indataset, censored)
        #Save with the noisy indata
        _savefile(filename + '_perfect_noisyinput' + extension, noisy_indataset, perfect)
        _savefile(filename + '_censored_noisyinput' + extension, noisy_indataset, censored)
        
    return (perfect, censored, noisy_indataset)
    
    
def _random_binaries(shape):
    '''Given a shape as (x, y) will return a numpy array of that shape with 1 and -1 randomly distributed in it.'''
    (rows, cols) = shape
    binarray = np.empty((0, cols), dtype=float)
    for _ in xrange(rows):
        binrow = []
        for _ in xrange(cols):
            #Select one of 1 and -1 to append to the list
            binrow.append(sample([1, -1], 1)[0])
        if cols > 1:
            binrow = [binrow]
        binarray = np.append(binarray, binrow, axis=0)
    
    return binarray
    
def add_noise_to(indataset, scale=1.5):
    '''Given a data set, this function will return the data set with some gaussian noise to every data point in it.
    Scale is the variable describing the likelihood of large changes. Higher values indicate larger probility for higher
    values.'''
    
    #This gives gaussian noise
    noise = np.random.exponential(scale, indataset.shape)
    #But it's only positive, so multiply some by -1
    binary = _random_binaries(indataset.shape)
    noise *= binary
    return indataset + noise
    
    