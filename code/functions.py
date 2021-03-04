import tensorflow as tf

def aggregation(input_, weight, kernel_size, stride, padding, dilation, pad_mode):
    kernel_size_unfold = [1, kernel_size, kernel_size, 1]
    strides_unfold = [1, stride, stride, 1]
    dilation_unfold = [1, dilation, dilation, 1]
    n, c_x, c_w, in_height, in_width = tf.shape(input_)[0], tf.shape(input_)[3], tf.shape(weight)[3], tf.shape(input_)[1], tf.shape(input_)[2]
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    unfold_j = tf.image.extract_patches(input_, kernel_size_unfold, strides_unfold, dilation_unfold, "SAME")
    
    out = tf.reshape(unfold_j, [n, c_x // c_w, c_w, pow(kernel_size, 2), out_height * out_width])
    hadamard = tf.math.reduce_sum(tf.expand_dims(weight, 1) * out, -2)
    hadamard = tf.reshape(hadamard, [n, c_x, out_height, out_width])
    return hadamard

class Aggregation(tf.keras.Model):
    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Aggregation, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def call(self, input_, weight):
        return aggregation(input_, weight, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
