from keras.activations import *

# dic is formated using the equation y = mx + b where the closest solution is described by (m, b)
LayerDictionary = {
    (type(relu), type(relu)): (1, 0),  # perfect transformation
    (type(tanh), type(tanh)): (1.1251, 0),
    (type(tanh), type(sigmoid)): (1.1843, 0),
    (type(sigmoid), type(tanh)): (1.1099, 0),
    (type(sigmoid), type(sigmoid)): (1.1677, 0)
}
