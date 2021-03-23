import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
x = np.array([10,9,8,7,6,5,4,3,2,1],dtype=float)
predict = x[len(x)-1]
y = np.array([i for i in x[1:]])
x = x[:len(x)-1]

model.fit(x,y,epochs=2000)
results = model.predict([[predict]])
rounded_results = [int(np.round(num)) for num in results]
print(results)
print(rounded_results)
