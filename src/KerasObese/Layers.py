from inspect import Signature
from typing import overload
from keras.layers import Dense
import numpy as np


class Layer:
    """
    A class to represent a keras layer

    ...

    Attributes
    ----------
    layer : None
        Copy of keras layer
    Type : type
        Type of layer currently stored
    Weights : list
        list of layer's weights
    activation : None
        Current layer's activation

    Methods
    -------
    buildLayer():
        Retrieves a copy of layer
    getWeights():
        Gets the layer's Weights
    getOutputShape():
        Gets the expected output shape. Note: Not always possible to get the output shape.
    """
    def __init__(self, layer):
        """Creates a Layer of unknown type. Note this layer is not modifiable.

        Args:
            layer (_type_): Layer to include in final build
        """
        self.layer = layer
        self.Type = type(layer)
        self.Weights = None
        self.activation = None
        try:
            self.outputShape = layer.output_shape
        except:
            self.outputShape = None

    def buildLayer(self):
        """Retrieves a copy of layer

        Returns:
            _type_: Returns the layer
        """
        return self.layer

    def getWeights(self) -> list:
        """Gets the layer's Weights

        Returns:
            list: Gets the layer's Weights
        """
        return self.Weights

    def getOutputShape(self) -> tuple:
        """Gets the expected output shape. Note: Not always possible to get the output shape.

        Returns:
            tuple: Output Shape
        """
        return self.outputShape


class DenseLayer(Layer):
    """
    A class to represent a keras Dense layer

    ...

    Attributes
    ----------
    layer : None
        Copy of keras layer
    Type : type
        Type of layer currently stored
    Weights : list
        list of layer's weights
    activation : None
        Current layer's activation

    Methods
    -------
    buildLayer():
        Creates keras layer from Weights
    getWeights():
        Gets the layer's Weights
    setWeights(Weights):
        Sets the layers weights
    getOutputShape():
        Gets the expected output shape. Note: Not always possible to get the output shape.
    """
    def __init__(self, layer: Dense, Weights: list = None):
        """Creates a clone of layer, optionally adds weights to layer.

        Args:
            layer (Dense): Layer to include in final build
            Weights (list, optional): Weights to use in the final build instead of default. Defaults to None.

        Raises:
            TypeError: Raised if layer is not an instance of Dense
            TypeError: Raised if Weights is not an instance of None or list
            ValueError: Raised if Weights does not have a length of 2
            TypeError: Raised Weights is not a list of numpy arrays
        """
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

    def setWeights(self, Weights: list):
        """Sets the layers weights

        Args:
            Weights (list): Weights to use in final build

        Raises:
            TypeError: Raised if Weights is not an instance of list
            ValueError: Raised if Weights does not have a length of 2
            TypeError: Raised Weights is not a list of numpy arrays
        """
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

    def buildLayer(self) -> Dense:  # Note: Cannot set weights before adding to model
        """Creates keras layer from Weights. Note: Does not return layer with Weights only layer of correct size for Weights

        Returns:
            Dense: Layer capable of receiving Weights once added to model.
        """
        units = np.shape(self.Weights[0])[1]  # Get units for new layer
        return Dense(units, activation=self.activation)
