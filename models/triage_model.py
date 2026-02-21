import tensorflow as tf
from config.config import *
from models.positional_encoding import positional_encoding
from models.transformer_block import TransformerBlock


def build_model(num_classes=3):
    inputs = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,))

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    )

    x = embedding_layer(inputs)
    x = x + positional_encoding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

    transformer = TransformerBlock(
        EMBEDDING_DIM,
        NUM_HEADS,
        FF_DIM
    )

    x = transformer(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)