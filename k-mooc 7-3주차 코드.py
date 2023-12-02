from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("train data(count,row,column): " +str(X_train.shape))
print("test data(count,row,column): " +str(X_test.shape))

print(X_train[0])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train[0])

print("train target (count): " +str(y_train.shape))
print("test target (count): " +str(y_test.shape))

print("sample from train: " +str(y_train[0]))
print("sample from test: " +str(y_test[0]))

input_dim = 784
X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(input_dim = input_dim, units = 10, activation = 'sigmoid'))
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 2048, epochs = 100, verbose = 0)

score = model.evaluate(X_test, y_test)
print(score[1])