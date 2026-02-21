import tensorflow as tf


def build_label_lookup(labels):
    return tf.keras.layers.StringLookup(
        vocabulary=sorted(set(labels)),
        output_mode="int"
    )