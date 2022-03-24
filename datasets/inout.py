import pandas as pd
import numpy as np

def csv2pd(path):
    return pd.read_table(path, sep=',', decimal='.', header=None)

def csv2np(path):
    return pd.read_table(path, sep=',', decimal='.', header=None).to_numpy()

def np2pd(data):
    return pd.DataFrame(data)

def pd2np(data):
    return data.to_numpy()

def np2csv(data, path):
    np.savetxt(path, data, delimiter=",")
    return path

def pd2csv(data, path):
    data.to_csv(path, columns=None, header=False, index=False)
    return path