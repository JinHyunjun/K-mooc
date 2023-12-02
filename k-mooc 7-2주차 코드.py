from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np

#단일 입력 로지스틱 회귀분석

model = Sequential()
model.add(Dense(input_dim=1, units = 1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd',
             metrics = ['binary_accuracy'])

X = np.array([-2,-1.5,-1,1.25,1.62,2])
Y = np.array([0, 0, 0, 1, 1, 1])

model.fit(X, Y, epochs = 300, verbose=0)

model.predict([-2,-1.5,-1,1.25,1.62,2])
model.predict([-1000, 1000])
model.summary()
model.layers[0].get_weights()

#다중 입력 로지스틱 회귀분석
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np

model = Sequential()
model.add(Dense(input_dim=2, units = 1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd',
             metrics = ['binary_accuracy'])

X = np.array([(0,0), (0,1), (1,0), (1,1)])
Y = np.array([0, 0, 0, 1])

model.fit(X, Y, epochs = 5000, verbose = 0)
model.predict(X)
model.summary()
model.layers[0].get_weights()