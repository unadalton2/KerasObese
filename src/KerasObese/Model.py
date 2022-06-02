from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras import activations
import numpy as np
from .Layers import DenseLayer
from .Dict import LayerDictionary


class Model:
    def __init__(self, model: Sequential) -> None:
        if not isinstance(model, Sequential):
            raise RuntimeError  # TODO add correct error
        self.oldModel = model
        self.inputShape = model.input_shape
        self.Layers = []

        for Layer in model.layers:
            self.Layers.append(DenseLayer(Layer))

    def AddLayer(self, index, activation=None):
        if not isinstance(index, int):
            raise RuntimeError("Expected index to be int instead got "+type(index).__name__)  # TODO add correct Error

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

        # TODO add M and C
        # Get weights ready
        newLayerWeights = [np.identity(newShape)[0]*M, np.zeros(newShape)[1]*B]

        # add weights to model
        self.Layers.insert(index+1, DenseLayer(Dense(newShape,
                           activation=newLayerActivation), newLayerWeights))
        # newlastLayer.set_weights(lastLayerWeights)#load weights
        #raise NotImplementedError

    def AddNeuron(self, index):
        if not isinstance(index, int):
            raise RuntimeError  # TODO add correct Error
        raise NotImplementedError

    def build(self):
        newModel = Sequential()
        newModel.add(InputLayer(self.inputShape))

        for Layer in self.Layers:
            newModel.add(Layer.buildLayer())

            newModel.layers[-1].set_weights(Layer.getWeights())

        return newModel
