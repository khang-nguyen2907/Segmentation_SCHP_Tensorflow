import functools
import math
from tensorflow.keras.layers import Conv2D, Layer, BatchNormalization, Activation, Add, MaxPool2D
from tensorflow.keras import Sequential
import tensorflow as tf

def conv3x3(in_, out_, stride, bias = False):
    """
    Create a 2D Convolutional layer
    :param in_ --  an Input tensor or a previous layer
    :param out_ --  the number of filters, it indicates the number of channels of the output Conv2d
    :param kernel_size --  the size of kernel sliding on Conv2d layer
    :param stride --
    :param bias --
    :return:
    """
    return Conv2D(filters=out_, kernel_size=3, strides=(stride, stride), padding="valid", use_bias=bias)(in_)

class BasicBlock(Layer):
    expansion = 1
    def __init__(self, in_, out_, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(filters=out_, kernel_size=3, strides=(stride,stride), padding='valid')
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters=out_, kernel_size=3, padding = 'valid')
        self.bn2 = BatchNormalization()

        self.act = Activation('relu')
        self.add = Add()

    def call(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(input_tensor)

        out += self.add([out, input_tensor])
        out = self.act(out)

        return out

class Bottleneck(Layer):
    expansion = 4

    def __init__(self, in_, out_, stride = 1, donwsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(filters=out_, kernel_size=1, strides=(1,1), padding='valid', use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=out_, kernel_size=3, strides=(stride,stride), padding='valid', use_bias=False)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(filters=out_*4, kernel_size=1, strides=(1, 1), padding='valid')
        self.bn3 = BatchNormalization()

        self.act = Activation('relu')
        self.add = Add()
        self.downsample = donwsample
        self.stride = stride

    def call(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out= self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if self.downsample is not None:
        #     residual = self.downsample(input_tensor)

        out = self.add([out, input_tensor])
        out = self.act(out)

        return out

class ResNet(Layer):
    def __init__(self, block, layers, num_classes=1000):
        self.in_ = 128
        super(ResNet, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=3,strides=(2,2), padding='valid', use_bias=False)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', use_bias=False)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid', use_bias=False)
        self.bn3 = BatchNormalization()

        self.act = Activation('relu')
        self.maxpool = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid')


    def _make_layer(self, block, out_, blocks, stride = 1 ):
        downsample = None
        if stride != 1 or self.in_ != out_ * block.expansion:
            downsample = Sequential([
                tf.keras.Input(shape = (self.in_,)),
                Conv2D(filters=out_*block.expansion, kernel_size=1, strides=(stride, stride), use_bias=False),
                BatchNormalization()
            ])







