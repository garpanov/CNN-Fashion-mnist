from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import keras
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train_new = x_train / 255
x_test_new = x_test / 255

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

x_train_new = np.expand_dims(x_train_new, axis=3)
x_test_new = np.expand_dims(x_test_new, axis=3)


model = keras.Sequential([
    Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(28,28,1)),
    MaxPooling2D((2,2), 2,),
    Conv2D(64, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2), 2,),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])


model.compile(optimizer="Adam", loss="binary_crossentropy")
model.fit(x_train_new, y_train, batch_size=35, epochs=5, validation_split=0.2)
nomer = ["футболка/топ", "брюки", "пуловера", "Платье",
         "пальто", "сандалий", "Рубашка", "кроссовок",
         "сумок", "Ботильоны"]
rez = model.predict(x_test_new[2:3])
rez = np.argmax(rez)
rez = nomer[rez]
print(rez)
x_test[2]
