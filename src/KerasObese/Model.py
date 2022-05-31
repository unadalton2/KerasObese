from keras.models import Sequential
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

    def AddLayer(index, activation=None):
        if not isinstance(index, int):
            raise RuntimeError  # TODO add correct Error
        raise NotImplementedError

    def AddNeuron(index):
        if not isinstance(index, int):
            raise RuntimeError  # TODO add correct Error
        raise NotImplementedError
