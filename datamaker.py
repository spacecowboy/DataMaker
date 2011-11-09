'''
This module is designed to create data sets for machine learning. Primarily
data intended to simulate censored survival data.
'''

def _makePerfectNoisy(function, in_data):
    '''y = f(x) +- noise
    Adds some noise to the otherwise perfect data.
    Returns a tuple, [y, event] where event will be 1 since this data will be
    uncensored.
    '''
    pass

def _makeCensored(function, in_data):
    '''y = f(x) - rand
    Same as perfect data except that the output will be censored, meaning
    y will be less than f(x). Less by some random value.
    y is guaranteed to be greater than zero.
    Returns a tuple, [y, event] where event is 0 because this data will be
    censored.
    '''
    pass

def _makeCensoredNoisy(function, in_data):
    '''y = f(x) - rand + noise
    Same as makeCensored but will then add some random noise as well. It adds
    the nosie after censoring to make sure that the censoring doesn't get rid
    of the noise.
    y guaranteed to be greater than zero
    Returns a tupe, [y, event] where event is 0 because this data is censored.
    '''
    pass

def _saveFile(filename, in_data_set, out_data_set):
    '''Will save the data set in a file with the name and path specified.
    '''
    pass

def createDataSets(function, in_data_set, censored_ratio = 0.5, filename = None):
    '''Given a function, and some in_data, this function will return a tuple, 
    [perfect, perfect_noisy, censored, censored_noisy]
    If given a filename, it will save the complete sets (including input) to
    files with the names: filename_perfect, filename_perfect_noisy,
    filename_censored, filename_censored_noisy (extension .txt if none
    specified).
    censored_ratio can be set to a value between 0 and 1 if a different amount
    of censoring is desired.
    Noise is set to 25% of the difference between lowest and highest output.
    '''
    pass