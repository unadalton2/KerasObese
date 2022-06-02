from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras import activations
import numpy as np
from .Layers import DenseLayer
from .Dict import LayerDictionary


class Model:
    def __init__(self, model: Sequential) -> None:
        """Creates a model capable of retaining pre-trained knowledge while changing network architecture. Currently only supports Sequential models.

        Args:
            model (Sequential): The model which modifications will be applied to

        Raises:
            TypeError: TypeError is raised when an invalid model is passed
        """
        if not isinstance(model, Sequential):
            raise TypeError  # TODO add correct error
        self.oldModel = model
        self.inputShape = model.input_shape
        self.Layers = []

        for Layer in model.layers:
            self.Layers.append(DenseLayer(Layer))

    def AddLayer(self, index: int, activation=None):
        """Adds a layer after layer at index. Maintains same size to previous layer size.

        Args:
            index (int): Specifies the layer to insert the new layer.
            activation (_type_, optional): The activation function the new layer will use. Defaults to previous layer's activation function.

        Raises:
            TypeError: _description_
        """
        if not isinstance(index, int):
            raise TypeError("Expected index to be int instead got "+type(index).__name__)  # TODO add correct Error

        oldLayerWeights = self.Layers[index].getWeights()  # Get Last layer
        oldLayerActivation = self.Layers[index].activation
        newLayerActivation = self.Layers[index].activation
        if not isinstance(activation, type(None)):
            if isinstance(activation, str):
                # Setting new layer activation from string to keras activation
                newLayerActivation = getattr(activations, activation)
            else:
                newLayerActivation = activation

        try:
            # Get the slope and Bias for activation cancellation
            M, B = LayerDictionary[(type(oldLayerActivation), type(newLayerActivation))]
        except:
            print(
                "Warning unknown combination of activation functions were found when creating layer "+str(index))
            M, B = 1, 0  # Fall back if unknown combination of functions were passed  into function

        # Calculate new shape for identity matrix
        newShape = np.shape(oldLayerWeights[0])[1]

        
        # Get weights ready
        newLayerWeights = [np.identity(newShape)[0]*M, np.zeros(newShape)[1]*B]

        # add weights to model
        self.Layers.insert(index+1, DenseLayer(Dense(newShape,
                           activation=newLayerActivation), newLayerWeights))
        # newLastLayer.set_weights(lastLayerWeights)#load weights
        #raise NotImplementedError

    def AddNeuron(self, index):
        if not isinstance(index, int):
            raise TypeError  # TODO add correct Error
        raise NotImplementedError

    def build(self):
        """Generates a working keras Model, currently only supporting Sequential

        Returns:
            _type_: Sequential
        """
        newModel = Sequential()
        newModel.add(InputLayer(self.inputShape))

        for Layer in self.Layers:
            newModel.add(Layer.buildLayer())

            newModel.layers[-1].set_weights(Layer.getWeights())

        return newModel
