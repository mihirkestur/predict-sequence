import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
def train_predict(sequence,n):
    x = np.array(sequence,dtype=float)
    predict = x[len(x)-1]
    y = np.array([i for i in x[1:]])
    x = x[:len(x)-1]
    model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer='sgd',loss='mean_squared_error')
    model.fit(x,y,epochs=1000)
    results = []
    for i in range(n):  
        prev = model.predict([predict])
        results.append(model.predict([predict]))
        predict = prev
    rounded_results = [int(float(np.round(num))) for num in results]
    #print(results)
    #print(rounded_results)
    #model.save("test.h5")
    #print(model.summary())
    return rounded_results

st.title("Sequence prediction")

seq = st.text_input("Enter sequence separated by commas (eg: 1,2,3)")
n_elem = int(st.text_input("Enter number of elements to be predicted"))
if(seq != ""):
    seq = [int(i) for i in seq.split(",")]
    result = train_predict(seq,n_elem)
    st.write("result is ",result)