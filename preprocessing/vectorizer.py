import tensorflow as tf
from config.config import MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH


def build_vectorizer(texts):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation"
    )
    vectorizer.adapt(texts)
    return vectorizer