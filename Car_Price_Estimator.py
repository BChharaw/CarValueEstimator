#imported packages are required to be installed for this program to work, including tensorflow, numpy
#This was a project I developed while doing the FreeCodeCamp Python Neural Networks for beginners course: https://www.freecodecamp.org/learn/machine-learning-with-python/ using a similar approach to defining the network as in the tutorial with a different dataset.
#re routing the path to the test.csv and train.csv files is required for the file to be loaded into the pandas dataframe

from __future__ import absolute_import, division, print_function, unicode_literals
import time
start = time.perf_counter()
import os
def clear ():
    os.system('cls')
clear()
print ("Running startup tasks")
#_______
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
#_______
clear()
end = time.perf_counter()
elapsed = math.ceil((end - start)*1000)/1000
print ("Startup took " + str(elapsed) + "s")

#loading dataset
TrainingDataframe = pd.read_csv('/Users/brenc/OneDrive/Desktop/Coding/TensorFlow/CarEstimator/train.csv') # training data
EvaluationDataframe = pd.read_csv('/Users/brenc/OneDrive/Desktop/Coding/TensorFlow/CarEstimator/test.csv') # testing data
TrSellPrice = TrainingDataframe.pop('sellprice')
EvSellPrice = EvaluationDataframe.pop('sellprice')


categoricalData = ['fuel', 'seller_type', 'transmission', 'owner']
numericData = ['year', 'km_driven', 'mileage(kmpl)','engine(CC)', 'maxpower(bhp)','seats']

feature_columns = []


for feature_name in categoricalData:
    vocab = TrainingDataframe[feature_name].unique() 
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))
#should be intended, creates a numpy array which references every item to a number

for feature_name in numericData:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype = tf.float32))

def inputfx (data, output, epochs=20, shuffle=True, batch = 32):
    def input():
        dataSetOBJ = tf.data.Dataset.from_tensor_slices((dict(data), output))
            #creating a dataset obj in a format tf can understand
        if shuffle:
            dataSetOBJ = dataSetOBJ.shuffle(1000) #shuffle
        dataSetOBJ = dataSetOBJ.batch(batch).repeat(epochs)
            #splits data into groups of size (batch), (epoch) num of times
        return dataSetOBJ
    return input

trainfx = inputfx(TrainingDataframe, TrSellPrice)
evalfx = inputfx(EvaluationDataframe, EvSellPrice, epochs = 1, shuffle=False)

model = tf.estimator.LinearRegressor(feature_columns=feature_columns)
model.train(trainfx) # training!! :)

results = model.evaluate(evalfx) #get info about testing the model
clear()

#ways of displaying results
PrintedRes = list(model.predict(evalfx))

for x in range(6):
    print("Actual Sell Price: " + str(math.ceil(EvSellPrice.loc[x]*10000)) + ". Model predicted price: " + str(math.ceil(PrintedRes[x]['predictions'][0]*10000)))
    print("Car data: " + str(EvaluationDataframe.loc[x]))
    print("________________________")


