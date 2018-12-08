import tensorflow as tf
import tensorflow_hub as hub


elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def ELMo(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]