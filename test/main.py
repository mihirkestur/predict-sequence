import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras

def train_predict(sequence):
    x = np.array(sequence,dtype=float)
    predict = x[len(x)-1]
    y = np.array([i for i in x[1:]])
    x = x[:len(x)-1]
    model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer='sgd',loss='mean_squared_error')
    model.fit(x,y,epochs=1000)
    results = model.predict([[predict],[5],[6]])
    rounded_results = [int(np.round(num)) for num in results]
    #print(results)
    #print(rounded_results)
    model.save("test.h5")
    return rounded_results
print(train_predict([1,2,3,4,5,6,7,8,9]))