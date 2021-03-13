import tensorflow as tf
import functions as F

class Subtraction2(tf.keras.Model):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction2, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def call(self, input1, input2):
        return F.subtraction2(input1, input2, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
