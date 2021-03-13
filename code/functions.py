import tensorflow as tf
import san

def aggregation(input_, weight, kernel_size, stride, padding, dilation, pad_mode):

    n, c_x, c_w, in_height, in_width = input_.shape[0], input_.shape[1], weight.shape[1], input_.shape[2], input_.shape[3]
    padding = (dilation * (kernel_size - 1) + 1) // 2
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    unfold_j = san.unfold(input_, kernel_size, dilation, padding, stride)
    out = tf.reshape(unfold_j, [n, c_x // c_w, c_w, pow(kernel_size, 2), out_height * out_width])

    hadamard = tf.math.reduce_sum(tf.expand_dims(weight, 1) * out, -2)
    hadamard = tf.reshape(hadamard, [n, c_x, out_height, out_width])
    return hadamard


def subtraction(input_, kernel_size, stride, padding, dilation, pad_mode):
    n, c, in_height, in_width = input_.shape[0], input_.shape[1], input_.shape[2], input_.shape[3]
    padding = (dilation * (kernel_size - 1) + 1) // 2
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    unfold_i = san.unfold(input_, 1, dilation, 0, stride)
    unfold_j = san.unfold(input_, kernel_size, dilation, padding, stride)
    y1 = tf.reshape(unfold_i,[n, c, 1, out_height * out_width])
    y2 = tf.reshape(unfold_j, [n, c, pow(kernel_size, 2), out_height * out_width])
    y = y1 - y2
    return y


def subtraction2(input1, input2, kernel_size, stride, padding, dilation, pad_mode):
    n, c, in_height, in_width = input1.shape[0], input1.shape[1], input1.shape[2], input1.shape[3]
    padding = (dilation * (kernel_size - 1) + 1) // 2
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    unfold_i = san.unfold(input1, 1, dilation, 0, stride)
    unfold_j = san.unfold(input2, kernel_size, dilation, padding, stride)
    y1 = tf.reshape(unfold_i,[n, c, 1, out_height * out_width])
    y2 = tf.reshape(unfold_j, [n, c, pow(kernel_size, 2), out_height * out_width])
    y = y1 - y2
    return y

