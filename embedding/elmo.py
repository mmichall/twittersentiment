import tensorflow as tf
import tensorflow_hub as hub


elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)