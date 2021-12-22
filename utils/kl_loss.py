import tensorflow as tf
from tensorflow.keras.losses import Loss, KLDivergence
import numpy as np

class KL(Loss):
    def __init__(self, ignored_index=255):
        super(KL, self).__init__()
        self.ignored_index = ignored_index
        self.kl = KLDivergence()

    def call(self, y_true, y_pred):
        """
        loss betweem parsing_pred  and soft_parsing_pred

        y_true -- parsing label or edge label -- b,h,w
        y_pred -- [parsing_pred, soft_parsing_pred] -- parsing_pred and soft_parsing_pred: b,h,w,c
        """
        C = y_pred[0].shape[3]
        inputs = tf.reshape(y_pred[0], (-1, C))  # (b*h*w,C)
        target = tf.reshape(y_pred[1], (-1, C))  # (b*h*w, C)
        label = tf.reshape(y_true, (-1))  # (b*h*w)

        valid = tf.not_equal(label, self.ignored_index)

        vy_pred = tf.boolean_mask(inputs, valid)
        vy_true = tf.boolean_mask(target, valid)

        loss = self.kl(vy_true, vy_pred)

        return loss


