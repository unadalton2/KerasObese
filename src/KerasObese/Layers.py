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
        if not isinstance(Weights, list):
            raise TypeError("Weights must be a list of numpy arrays instead got: "+type(Weights).__name__)
        if len(Weights) != 2:
            raise ValueError("Weights must be a list with length 2 instead got: "+ len(Weights))
        for i in range(len(Weights)):
            if not isinstance(Weights[i], np.ndarray):
                raise TypeError("Weights must be a list of only numpy arrays found: "+type(Weights[i]).__name__+" at position: "+str(i))
        self.Weights = Weights
            
            
    
    def buildLayer(self): # Note: Cannot set weights before adding to model
        #print(np.shape(self.Weights[0]))
        units = np.shape(self.Weights[0])[1]#Get units for new layer
        return Dense(units, activation=self.activation)
