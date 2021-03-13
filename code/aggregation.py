import tensorflow as tf
import functions as F

class Aggregation(tf.keras.Model):
    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Aggregation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def call(self, input_, weight):
        return F.aggregation(input_, weight, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
