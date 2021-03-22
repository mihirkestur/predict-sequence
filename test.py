import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
data = np.array(data, dtype=np.float32)
target = [(i+5)/100 for i in range(100)]
target = np.array(target, dtype=np.float32)
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size = 0.2,random_state=4)

model = Sequential()  
model.add(LSTM((1),batch_input_shape=(None,5,1),return_sequences=False))
#model.add(Dense(100))
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
his = model.fit(x_train, y_train, epochs=400, batch_size=1, verbose=2,validation_data=(x_test, y_test))
print(model.predict(np.array([[[0.01],[0.02],[0.03],[0.04],[0.05]]], dtype=np.float32))*100)

"""
res = model.predict(x_test)
plt.scatter(range(20),res,c='r')
plt.scatter(range(20),y_test,c='g')
plt.show()
plt.plot(his.history['loss'])
plt.show()
"""