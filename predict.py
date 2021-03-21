from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers.recurrent import LSTM

layers = [LSTM(2), Dense(1)]
model = Sequential(layers)
model = Sequential()
model.add(LSTM(5, input_shape=(2,1)))
model.add(Dense(1))
model = Sequential()
model.add(LSTM(5, input_shape=(2,1)))
model.add(Dense(1))
model.add(Activation("sgd"))
model.compile(optimizer= sgd , loss= mse )
algorithm = SGD(lr=0.1, momentum=0.3)
model.compile(optimizer=algorithm, loss= mse )
model.compile(optimizer= sgd , loss= mean_squared_error , metrics=[ accuracy ])
model.fit(X, y, batch_size=32, epochs=100)
history = model.fit(X, y, batch_size=10, epochs=100, verbose=0)
loss, accuracy = model.evaluate(X, y)
loss, accuracy = model.evaluate(X, y, verbose=0)
predictions = model.predict(X)
predictions = model.predict_classes(X)
predictions = model.predict(X, verbose=0)