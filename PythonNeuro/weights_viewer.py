import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.load_model("mnist_model.h5")

for layer in model.layers:
    weights = layer.get_weights()
    print(layer.name)
    print(weights)
