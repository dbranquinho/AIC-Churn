import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_keras_model(input_dim, hidden_units=[128, 64, 32], dropout_rate=0.3):
    """
    Builds a Keras Sequential MLP equivalent to the PyTorch ChurnModel.
    
    Architecture mirrors the PyTorch model:
        Dense -> BatchNorm -> ReLU -> Dropout (for each hidden layer)
        Dense(1, sigmoid) output for binary classification
    
    Args:
        input_dim (int): Number of input features.
        hidden_units (list of int): Sizes of hidden layers.
        dropout_rate (float): Dropout probability.
        
    Returns:
        A compiled tf.keras.Model.
    """
    model = keras.Sequential(name="ChurnModel_TF")
    model.add(layers.InputLayer(shape=(input_dim,)))
    
    for units in hidden_units:
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer with sigmoid for binary classification probability
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
