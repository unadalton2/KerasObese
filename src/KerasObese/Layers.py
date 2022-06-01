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
    def __init__(self, layer: Dense = None, Weights: list = None):
        if not isinstance(layer, Dense):
            raise RuntimeError  # TODO add correct error
        if not isinstance(layer, type(None)):
            super().__init__(layer)
            self.Weights = layer.get_weights()
        else:
            if not isinstance(Weights, type(None)):
                # TODO create Dense from weights
                super().__init__(Dense())
            else:
                raise RuntimeError  # TODO add Correct Error
        if not isinstance(Weights, type(None)):
            self.Weights = Weights

        self.activation = self.layer.activation
    """
    def __init__(self, Type: str, Weights: np.array):
        raise NotImplementedError
        super().__init__(layer)
        # TODO Add type Checking
        self.Weights = layer.get_weights()
    """

    def setWeights(self, Weights):
        # TODO Add type Checking
        raise NotImplementedError
    
    def buildLayer(self): # Note: Cannot set weights before adding to model
        units = np.shape(self.Weights[0])[1]#Get units for new layer
        return Dense(units, activation=self.activation)
