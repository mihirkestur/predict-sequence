import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras

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

app = Flask(__name__)
#model = keras.models.load_model('test.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    seq = request.form.get("Sequence")
    next_n = int(request.form.get("next"))
    seq = [int(float(i)) for i in seq.split(",")]
    result = train_predict(seq,next_n)
    #seq = np.array(seq,dtype=float)
    #results = model.predict([seq])
    #rounded_results = [int(np.round(num)) for num in results]
    return render_template('index.html', prediction_text='Next {} numbers in the sequence {} is : {}'.format(next_n,seq,result))

if __name__ == "__main__":
    app.run(debug=True)