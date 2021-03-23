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
his = model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2,validation_data=(x_test, y_test))
#print(model.predict(np.array([[[10.00],[10.01],[10.02],[10.03],[10.04]]], dtype=np.float32))*100)
#print(model.predict(x_test))
res = model.predict(x_test)
plt.scatter(range(20),res,c='r')
plt.scatter(range(20),y_test,c='g')
plt.show()
plt.plot(his.history['loss'])
plt.show()
print(model.predict_classes(np.array([[[1],[2],[3],[4],[5]]], dtype=np.float32)))