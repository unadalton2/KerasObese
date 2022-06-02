from keras.activations import *

# dic is formated using the equation y = mx + b where the closest solution is described by (m, b)
LayerDictionary = {
    (relu.__name__, relu.__name__): (1, 0),  # perfect transformation
    (tanh.__name__, tanh.__name__): (1.1251, 0),
    (tanh.__name__, sigmoid.__name__): (1.1843, 0),
    (sigmoid.__name__, tanh.__name__): (1.1099, 0),
    (sigmoid.__name__, sigmoid.__name__): (1.1677, 0)
}
