import tensorflow as tf
import san

def aggregation(input_, weight, kernel_size, stride, padding, dilation, pad_mode):
    kernel_size_unfold = [1, kernel_size, kernel_size, 1]
    strides_unfold = [1, stride, stride, 1]
    dilation_unfold = [1, dilation, dilation, 1]
    n, c_x, c_w, in_height, in_width = input_.shape[0], input_.shape[1], weight.shape[1], input_.shape[2], input_.shape[3]
    padding = (dilation * (kernel_size - 1) + 1) // 2
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    # print("out_width, out_height", out_width, out_height)
    # print(input_.shape)
    # unfold_j = tf.image.extract_patches(input_, kernel_size_unfold, strides_unfold, dilation_unfold, "SAME")
    unfold_j = san.unfold(input_, kernel_size, dilation, padding, stride)
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

def subtraction(input_, kernel_size, stride, padding, dilation, pad_mode):
    pass


class Subtraction(tf.keras.Model):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode

    def forward(self, input_):
        return subtraction(input_, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
