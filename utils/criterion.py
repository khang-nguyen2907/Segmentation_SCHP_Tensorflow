import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer, UpSampling2D
from utils.consistency_loss import ConsistencyLoss
from utils.lovasz_softmax import LovaszSoftmax
from tensorflow.keras.losses import Loss
from utils.cross_entropy import CrossEntropy
from utils.kl_loss import KL

class CriterionAll(Loss):
    def __init__(self, use_class_weight = False, ignore_index = 255, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1,
                 num_classes = 19):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.use_class_weight = use_class_weight
        self.criterion = CrossEntropy(ignored_index=ignore_index)
        self.lovasz = LovaszSoftmax(ignore_index = ignore_index)
        self.kldiv = KL(ignored_index=ignore_index)
        self.reg = ConsistencyLoss(ignore_index=ignore_index)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.num_classes = num_classes

    def parsing_loss(self, preds, target, cycle_n = None):
        """

        :param preds -- [[parsing_result, fusion_result], [edge_result]]
                        parsing_result, fusion_result -- [b, h, w, c]
                        edge_result                   -- [b, h, w, 2]
            **Note**    h and w of parsing_result, fusion_result is not equal to the original image
                        not equal to the size of label. For example: , if images are [b, 3, 473, 473],
                        labels are [b, 473, 473], edges are [b, 473, 473]. Then, parsing_result is
                        [b, c, 119, 119], fusion_result is [b, c, 119, 119]

        :param target -- [parsing_labels, edges_label, soft_preds, soft_edges]
            **Note**    soft_preds just contains fusion_result
        :param cycle_n:
        :return:
        """
        h, w = target[0].shape[1], target[0].shape[2] # target[0] is "parsing_labels"

        # pos_num = tf.reduce_sum(tf.cast(tf.where(target[1] == 1, target[1])))
        # neg_num = tf.reduce_sum(tf.cast(tf.where(target[1] == 0, target[1])))

        # weight_pos = neg_num / (pos_num + neg_num)
        # weight_neg = pos_num / (pos_num + neg_num)
        # weights = tf.convert_to_tensor([weight_neg, weight_pos])

        loss = 0

        #loss for segmentation
        preds_parsing = preds[0] # parsing_result, fusion_result
        for pred_parsing in preds_parsing:
            scale_pred = tf.compat.v1.image.resize_bilinear(pred_parsing, size = [h,w], align_corners=True)
            loss += 0.5 * self.lambda_1 * self.lovasz(target[0],scale_pred)
            # print("Lovazs loss:", 0.5 * self.lambda_1 * self.lovasz(target[0],scale_pred))

            if target[2] is None: #target[2] = soft_preds
                loss += 0.5 * self.lambda_1 * self.criterion(target[0], scale_pred)
                # print("Seg cross_entropy:", self.lambda_1 * self.criterion(target[0], scale_pred))
            else:
                soft_scale_pred = tf.compat.v1.image.resize_bilinear(target[2], size = [h,w], align_corners=True)
                soft_scale_pred = moving_average(soft_scale_pred, to_one_hot(target[0], self.num_classes), 1.0 / (cycle_n + 1.0))
                loss += 0.5 * self.lambda_1 * self.kldiv(target[0],[scale_pred, soft_scale_pred])

        #loss for edge
        preds_edge = preds[1]
        for pred_edge in preds_edge:
            scale_pred = tf.compat.v1.image.resize_bilinear(pred_edge, size=[h,w],align_corners=True )

            if target[3] is None:
                loss += self.lambda_2 * self.criterion(target[1], scale_pred)
                # print("Edge cross loss: ",self.lambda_2 * self.criterion(target[1], scale_pred))
            else:
                soft_scale_edge = tf.compat.v1.image.resize_bilinear(target[3], size=[h,w], align_corners=True)
                soft_scale_edge = moving_average(soft_scale_edge, to_one_hot(target[1], 2), 1.0/(cycle_n+1.0))
                loss += self.lambda_2 * self.kldiv(target[0],[scale_pred, soft_scale_edge])

        #consistency regularization
        preds_parsing = preds[0]
        preds_edge = preds[1]
        for pred_parsing in preds_parsing:
            scale_pred = tf.compat.v1.image.resize_bilinear(pred_parsing, size=[h,w], align_corners=True)

            scale_edge = tf.compat.v1.image.resize_bilinear(preds_edge[0], size=[h,w], align_corners=True)
            pred = [scale_pred, scale_edge]
            loss += self.lambda_3 * self.reg(target[0], pred)
            # print("Consistency Loss: ", self.lambda_3 * self.reg(target[0], pred))

        return loss


    def call(self, target, preds):
        pred, cycle_n = preds[0], preds[1]
        loss = self.parsing_loss(pred, target, cycle_n)
        return loss

def moving_average(target1, target2, alpha=1.0):
    target = 0
    target += (1.0 - alpha)*target1
    target += target2 * alpha
    return target

def to_one_hot(tensor, num_cls):
    """
    encode mask from (b,h,w) to (b, h, w, c) with c is the number of classes
    :param tensor -- mask with dimension (b, h, w)
    :param num_cls -- number of parsing class
    :return:
        an encoded mask (label) with dimension (b, h, w, c)
    """
    tensor = tf.cast(tensor, dtype=tf.int64)
    onehot_tensor = tf.one_hot(tensor, num_cls)
    return onehot_tensor
