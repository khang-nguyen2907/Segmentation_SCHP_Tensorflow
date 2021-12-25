import tensorflow as tf
from tensorflow.keras.losses import Loss, KLDivergence
import numpy as np

def flatten_probas(input, target, labels, ignore_index = 255):
    C = input.shape[3]
    input = tf.reshape(input, (-1, C))
    target = tf.reshape(target, (-1, C))
    labels = tf.reshape(labels, (-1))
    mask = (labels==ignore_index)
    vinput = tf.boolean_mask(input, mask)
    vtarget = tf.boolean_mask(target, mask)
    return vinput, vtarget

class KL(Loss):
    def __init__(self, ignored_index = 255):
        super(KL, self).__init__()
        self.ignored_index = ignored_index
        self.kl = KLDivergence()
        self.T = 1

    def call(self, labels, inputs):
        """
        inputs = [input, target]
        """
        input = inputs[0]
        target = inputs[1]
        log_input_prob = tf.nn.log_softmax(input / self.T, axis=3)
        target_prob = tf.nn.softmax(target / self.T, axis=3)
        vinput, vtarget = flatten_probas(log_input_prob, target_prob, labels)
        loss = self.kl(vtarget,vinput)
        return loss
