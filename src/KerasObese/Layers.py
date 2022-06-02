from inspect import Signature
from typing import overload
from keras.layers import Dense
import numpy as np


class Layer:
    def __init__(self, layer):
        self.layer = layer
        self.Type = type(layer)
        self.Weights = None
        self.activation = None
        try:
            self.outputShape = layer.output_shape
        except:
            self.outputShape = None

    def buildLayer(self):
        return self.layer

    def getWeights(self):
        return self.Weights

    def getOutputShape(self):
        return self.outputShape


class DenseLayer(Layer):
    def __init__(self, layer: Dense, Weights: list = None):
        if not isinstance(layer, Dense):
            raise TypeError(
                "layer must be a dense layer, instead got "+type(layer).__name__)
        super().__init__(layer)
        self.Weights = layer.get_weights()

        if not isinstance(Weights, type(None)):
            if not isinstance(Weights, list):
                raise TypeError(
                    "Weights must be a list of numpy arrays instead got: "+type(Weights).__name__)
            if len(Weights) != 2:
                raise ValueError(
                    "Weights must be a list with length 2 instead got: " + len(Weights))
            for i in range(len(Weights)):
                if not isinstance(Weights[i], np.ndarray):
                    raise TypeError("Weights must be a list of only numpy arrays found: " +
                                    type(Weights[i]).__name__+" at position: "+str(i))
            self.Weights = Weights

        self.activation = self.layer.activation

    def setWeights(self, Weights):

        if not isinstance(Weights, list):
            raise TypeError(
                "Weights must be a list of numpy arrays instead got: "+type(Weights).__name__)
        if len(Weights) != 2:
            raise ValueError(
                "Weights must be a list with length 2 instead got: " + len(Weights))
        for i in range(len(Weights)):
            if not isinstance(Weights[i], np.ndarray):
                raise TypeError("Weights must be a list of only numpy arrays found: " +
                                type(Weights[i]).__name__+" at position: "+str(i))
        self.Weights = Weights

    def buildLayer(self):  # Note: Cannot set weights before adding to model
        # print(np.shape(self.Weights[0]))
        units = np.shape(self.Weights[0])[1]  # Get units for new layer
        return Dense(units, activation=self.activation)
