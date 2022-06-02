from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras import activations
import numpy as np
from .Layers import DenseLayer, Layer
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

        for L in model.layers:
            if isinstance(L, Dense):
                self.Layers.append(DenseLayer(L))
            else:
                self.Layers.append(Layer(L))

    def AddLayer(self, index: int, activation=None):
        """Adds a layer after layer at index. Maintains same size to previous layer size.

        Args:
            index (int): Specifies the layer to insert the new layer.
            activation (_type_, optional): The activation function the new layer will use. Defaults to previous layer's activation function.

        Raises:
            TypeError: _description_
        """
        if not isinstance(index, int):
            # TODO add correct Error
            raise TypeError(
                "Expected index to be int instead got "+type(index).__name__)

        oldLayerWeights = self.Layers[index].getWeights()  # Get Last layer
        oldLayerActivation = self.Layers[index].activation
        newLayerActivation = self.Layers[index].activation
        if not isinstance(activation, type(None)):
            if isinstance(activation, str):
                # Setting new layer activation from string to keras activation
                try:
                    newLayerActivation = getattr(activations, activation)
                except:
                    raise ValueError(
                        "activation must be a valid child from keras.activations instead got: "+activation)
                    
                    
            else:
                newLayerActivation = activation

        try:
            # Get the slope and Bias for activation cancellation
            M, B = LayerDictionary[(
                oldLayerActivation.__name__, newLayerActivation.__name__)]
        except:
            print(
                "Warning unknown combination of activation functions were found when creating layer "+str(index))
            M, B = 1, 0  # Fall back if unknown combination of functions were passed  into function

        # Calculate new shape for identity matrix
        newShape = np.shape(oldLayerWeights[0])[1]

        # Get weights ready
        newLayerWeights = [np.identity(newShape)*M, np.zeros(newShape)+B]

        # add weights to model
        self.Layers.insert(index+1, DenseLayer(Dense(newShape,
                           activation=newLayerActivation), newLayerWeights))
        # newLastLayer.set_weights(lastLayerWeights)#load weights
        #raise NotImplementedError

    def AddNeuron(self, index: int):
        """Adds a neuron to layer at index

        Args:
            index (int): Layer to add neuron to

        Raises:
            TypeError: Raised when index is not an int
            ValueError: Raised when index is less then 0
        """
        if not isinstance(index, int):
            raise TypeError(
                "Expected index to be int instead got "+type(index).__name__)
        if index < 0:
            raise ValueError("index must be >= 0")

        # Gets Params from layer at index
        l1OldParams = self.Layers[index].getWeights()
        l1OldWeights = l1OldParams[0]
        l1OldBias = l1OldParams[1]

        # Getting Params for next layer
        if(index+1 != len(self.Layers)):
            l2OldParams = self.Layers[index+1].getWeights()
            l2OldWeights = l2OldParams[0]

        # Gets sizes needed for reshaping
        prevLayerSize = np.shape(self.Layers[index].getWeights()[0])[0]
        newSize = np.shape(l1OldWeights)[1]+1

        # Padding Weights with random numbers np.zeros
        l1NewWeights = np.random.uniform(-1, 1, (prevLayerSize, newSize))
        l1NewWeights[:, :-1] = l1OldWeights

        # Padding Layer 1 Bias with zero
        l1NewBias = np.pad(l1OldBias, (0, 1), 'constant')

        # Modify next layer (Ignore if index is currently on the last layer)
        if index+1 != len(self.Layers):
            # Get next layer output size to maintain size
            nextLayerOutputSize = np.shape(
                self.Layers[index+1].getWeights()[0])[1]

            # Padding with zeros so new neuron doesn't effect model output
            l2NewWeights = np.zeros((newSize, nextLayerOutputSize))

            l2NewWeights[:-1, :] = l2OldWeights

            ##nextLayerNewWeights = np.zeros((newSize, nextLayerOutputSize))
            l2OldParams[0] = l2NewWeights

        # setting new layer weights
        self.Layers[index].setWeights([l1NewWeights, l1NewBias])
        if index+1 != len(self.Layers):
            self.Layers[index+1].setWeights(l2OldParams)

        #raise NotImplementedError

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
