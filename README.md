KerasObese is a simple utility for altering the architecture of pre trained keras models without losing previously learned knowledge. KerasObese does not freeze any weights, instead KerasObese works by adding neurons and layers to the keras model in a way that mitigates the difference between the original and new model.

Simple Example:
```
import keras
from keras.layers import Dense
from keras.activations import relu, tanh
import numpy as np
import KerasObese as ko

#Creating Keras Model
TFmodel = keras.Sequential()
TFmodel.add(keras.layers.Dense(10, relu))
TFmodel.add(keras.layers.Dense(5, relu))
TFmodel.add(keras.layers.Dense(2, tanh))

print("compile")
TFmodel.compile(loss=keras.losses.MeanSquaredError())

#Training Keras model
X = np.ones((2,10))
Y = np.ones((2,2))
TFmodel.fit(X, Y, epochs=1)

data = np.ones((1, 10))
TFmodel.predict(data)
print("\n\n")


print("Old Model")
TFmodel.summary()

print("Modifying Keras model")
KOModel = ko.Model(TFmodel)
KOModel.AddDenseLayer(0, relu)
KOModel.AddDenseLayer(0)
KOModel.AddNeuron(0)

#Creating New Model with Modifications
newModel = KOModel.build()

print("New Model")
newModel.summary()

#Calculating diffrences
print()
print(TFmodel.predict(data))
print(newModel.predict(data))
print()
print("Total Difference: " + str(np.abs(TFmodel.predict(data)-newModel.predict(data))))
```