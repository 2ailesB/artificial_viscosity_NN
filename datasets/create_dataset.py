from cmath import pi
import numpy as np

from datasets.inout import np2csv
from datasets.fis import f1, f2, f3, f4, f5
from datasets.fourriercollocation import compute_fourier, select_stencil

def compute_fis(function, x, parameters):
    fs = np.zeros((len(parameters), len(x)))
    four = np.zeros((len(parameters), len(x)))
    for i, parameter in enumerate(parameters):
        fs[i, :] = function(x, parameter) #(nbparams, 401)
        four[i, :] = compute_fourier(function(x, parameter)) #(nbparams, 401)
    return fs, four

def prepare_f1(parameters):
    x = np.linspace(0, 2*pi, 401)
    fs, _ = compute_fis(f1, x, parameters) # evaluate function for all sets of parameters
    D = ((x >= 0) & (x <= 2*pi)) # compute domain for sampling
    fs = fs[:, D] # reduce function to the domain
    ech = select_stencil(fs, 7) #(nb params, 7)
    return np.concatenate((ech, 3 * np.ones((ech.shape[0], 1))), axis = 1)

def prepare_f2(parameters):
    x = np.linspace(0, 2*pi, 401)
    fs, _ = compute_fis(f2, x, parameters)
    D = ((x >= 3.53) & (x <= 5.89))
    fs = fs[:, D] # reduce function to the domain
    ech = select_stencil(fs, 7)
    return np.concatenate((ech, 3 * np.ones((ech.shape[0], 1))), axis = 1)

def prepare_f345(parameters, function, label):
    x = np.linspace(0, 2*pi, 401)
    fs, _ = compute_fis(function, x, parameters)
    ech = np.zeros((fs.shape[0], 7))
    for i, parameter in enumerate(parameters):
        _, _, a3 = parameter[0], parameter[1], parameter[2] ### TODO
        D = ((x >= a3 - 0.05) & (x <= a3 + 0.05))
        if D.sum() <= 7:
            D = ((x >= a3 - 0.05) & (x <= a3 + 0.06)) # add some points in D
        cur_fs = np.expand_dims(fs[i, D], 0)# reduce function to the domain
        ech[i, :] = select_stencil(cur_fs, 7)
    return np.concatenate((ech, label * np.ones((ech.shape[0], 1))), axis = 1)

def create_dataset():
    x = np.linspace(0, 2*pi, 401)
    af1 = np.arange(-20, 20, 0.5)
    af2 = np.arange(-10, 11, 1)
    af345 = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10, 10, 1), np.arange(0.25, 2.6, 0.25)) # TODO : do a grid
    af345 = np.concatenate((np.expand_dims(af345[0].flatten(), 1), np.expand_dims(af345[1].flatten(), 1), np.expand_dims(af345[2].flatten(), 1)), axis=1)
    f1s = prepare_f1(af1)
    f2s = prepare_f2(af2)
    f3s = prepare_f345(af345, f3, 0)
    f4s = prepare_f345(af345, f4, 1)
    f5s = prepare_f345(af345, f5, 2)

    dataset = np.concatenate((f1s, f2s, f3s, f4s, f5s), axis=0)

    return dataset


    