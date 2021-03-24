import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
app = Flask(__name__)
model = tf.keras.models.load_model('test.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    seq = int(request.form.get("Sequence"))
    #seq = [int(i) for i in seq.split(",")]
    #seq = np.array(seq,dtype=float)
    results = model.predict([seq])
    rounded_results = [int(np.round(num)) for num in results]
    return render_template('index.html', prediction_text='Next number in sequence is : {}'.format(rounded_results))

if __name__ == "__main__":
    app.run(debug=True)