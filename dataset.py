from sklearn import datasets
from datasets.create_dataset import create_dataset
from datasets.inout import np2csv

dataset = create_dataset()
np2csv(dataset, 'data/custom_data.csv')
