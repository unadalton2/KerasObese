from keras.models import Sequential
from keras.layers import InputLayer
from keras import activations
from numpy import isin
from .Layers import *


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
            raise RuntimeError  # TODO add correct Error
        if not ( isinstance(activation, type(None)) or isinstance(activation, str) or isinstance(activation, activations) ):
            raise RuntimeError # TODo add correct Error

        oldLayerWeights = self.Layers[index].getWeights()#Get Last layer
        newLayerActivation = self.Layers[index].activation
        if not isinstance(activation, type(None)):
            newLayerActivation = activation
        newShape = np.shape(oldLayerWeights[0])[1]#Calculate new shape for identity matrix

        # TODO add M and C
        newLayerWeights = [np.identity(newShape), np.zeros(newShape)] #Get weights ready

        self.Layers.insert(index+1, DenseLayer(Dense(newShape, activation=newLayerActivation), newLayerWeights))#add weights to model
        #newlastLayer.set_weights(lastLayerWeights)#load weights
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
