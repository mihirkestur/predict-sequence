import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
x = np.array([1,2,3,4,5],dtype=float)
y = np.array([25,50,75,125,150])
model.fit(x,y,epochs=3000)
results = model.predict([6])
rounded_results = [int(np.round(num)) for num in results]
print(results)
print(rounded_results)