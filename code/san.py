import tensorflow as tf 
from config import NUM_CLASSES
import math
import functions
from functions import Aggregation

'''
tensorflow 4-D tensor -> (batchsize, width, height, channels) 
torch 4-D tensor -> (batchsize, channels, height, width) 

MY RUNNING CONFIGURATION:
Tensorflow --> data format = channels first (batchsize, channels, height, width)

'''

def conv_1x1(out_dim, stride=1):
    return tf.keras.layers.Conv2D(filters=out_dim,
                            kernel_size=(1,1),
                            strides=stride,
                            padding="same",
                            data_format="channels_first",
                            use_bias=False) 

def position(width, height):
    repeat_multiples = tf.constant([height, 1], tf.int32)
    loc_w = tf.tile(tf.expand_dims(tf.linspace(-1.0, 1.0, width), axis=0), repeat_multiples)
    repeat_multiples = tf.constant([1, width], tf.int32)
    loc_h = tf.tile(tf.expand_dims(tf.linspace(-1.0, 1.0, height), axis=1), repeat_multiples)
    loc = tf.expand_dims(tf.concat([tf.expand_dims(loc_w, 0), tf.expand_dims(loc_h, 0)], 0),0)
    return loc

def unfold(input_, kernel_size, dilation, padding, stride):
    dim2 = (pow(kernel_size, 2)) * input_.shape[1]
    dim3 = int(pow((input_.shape[2] + 2 * padding - (dilation * (kernel_size - 1) + 1))/stride + 1, 2))
    out = tf.image.resize(input_, [dim2, dim3])
    out = out[:,:,:,0]
    return out


class SelfAttention(tf.keras.Model):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, strides=1, dilation=1):
        super(SelfAttention, self).__init__()
        self.sa_type, self.kernel_size, self.strides, self.dilation = sa_type, kernel_size, strides, dilation
        self.out_planes, self.share_planes = out_planes, share_planes
        self.conv1 = conv_1x1(rel_planes)
        self.conv2 = conv_1x1(rel_planes)
        self.conv3 = conv_1x1(out_planes)
        if sa_type == 0:
            self.conv_w = tf.keras.Sequential([tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            conv_1x1(rel_planes),
                                            tf.keras.layers.BatchNormalization(),
                                            tf.keras.layers.ReLU(),
                                            conv_1x1(out_planes//share_planes)])
            # self.bn_w1 = tf.keras.layers.BatchNormalization()
            # self.relu_w1 = tf.keras.layers.ReLU()
            # self.conv_w1 = conv_1x1(rel_planes)
            # self.bn_w2 = tf.keras.layers.BatchNormalization()
            # self.relu_w2 = tf.keras.layers.ReLU()
            # self.conv_w2 = conv_1x1(out_planes//share_planes)
            self.conv_p = tf.keras.layers.Conv2D(2, kernel_size=(1,1))
            # self.subtraction = Subtraction(kernel_size, strides, (dilation * (kernel_size-1)+1)//2, dilation, pad_mode=1)
            # self.subtraction = Subtraction2(kernel_size, strides, (dilation * (kernel_size-1)+1)//2, dilation, pad_mode=1)
            self.softmax = tf.keras.layers.Softmax(axis=-2)
        else:
            self.bn_w1 = tf.keras.layers.BatchNormalization()
            self.relu_w1 = tf.keras.layers.ReLU()
            self.conv_w1 = conv_1x1(out_planes//share_planes)
            self.bn_w2 = tf.keras.layers.BatchNormalization()
            self.relu_w2 = tf.keras.layers.ReLU()
            self.conv_w2 = conv_1x1(pow(kernel_size, 2)*out_planes//share_planes)
        self.aggregation = Aggregation(kernel_size, strides, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
    
    def call(self, input_):
        x1, x2, x3 = self.conv1(input_), self.conv2(input_), self.conv3(input_)
        if self.sa_type == 0:
            
            repeat = tf.constant([tf.shape(input_)[0], 1, 1, 1], tf.int32)
            p = self.conv_p(position(input_.shape[1], input_.shape[2]))
            w = self.softmax(self.conv_w(tf.concat([self.subtraction2[x1, x2], tf.tile(self.subtraction(p), repeat)], 1)))
        else:
            # kernel_size_unfold_i = [1, 1, 1, 1]
            # kernel_size_unfold_j = [1, self.kernel_size, self.kernel_size, 1]
            # strides_unfold = [1, self.strides, self.strides, 1]
            # dilation_unfold = [1, self.dilation, self.dilation, 1]
            padding = tf.constant([[0, 0], [0, 0], 
                                [self.kernel_size//2, self.kernel_size//2], 
                                [self.kernel_size//2, self.kernel_size//2]])

            if self.strides !=1:
                x1 = unfold(x1, 1, self.dilation, 0, self.strides) 
                # x1 = tf.image.extract_patches(x1, kernel_size_unfold_i, strides_unfold, dilation_unfold, "SAME")
           
            x1 = tf.reshape(x1, [tf.shape(input_)[0], -1, 1, input_.shape[2]*input_.shape[3]])
            pad = tf.pad(x2, padding, "REFLECT")
            # unfold_j = tf.image.extract_patches(pad, kernel_size_unfold_j, strides_unfold, dilation_unfold, "SAME")
            unfold_j = unfold(pad, self.kernel_size, self.dilation, 0, self.strides) 
            x2 = tf.reshape(unfold_j, [tf.shape(input_)[0], -1, 1, x1.shape[-1]])

            out = tf.concat([x1, x2], 1)
            out = self.bn_w1(out)
            out = self.conv_w1(self.relu_w1(out))
            out = self.conv_w2(self.relu_w2(self.bn_w2(out)))
            w = tf.reshape(out, [tf.shape(input_)[0], -1, pow(self.kernel_size, 2), x1.shape[-1]])
        x = self.aggregation(x3, w)
        return x

class Transition(tf.keras.Model):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, strides=1):
        super(Transition, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.sab = SelfAttention(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, strides)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv = conv_1x1(out_planes)
        self.relu = tf.keras.layers.ReLU()
        self.strides = strides
    
    def call(self, input_):
        identity = input_
        out = self.relu(self.bn1(input_))
        out = self.relu(self.bn2(out))
        out = self.relu(self.bn2(self.sab(out)))
        out = self.conv(out)
        out += identity
        return out


class SAN(tf.keras.Model):
    def __init__(self, sa_type, block, layers, kernels):
        super(SAN, self).__init__()
        channel = 64
        self.inital_conv = conv_1x1(channel)
        self.initial_bn = tf.keras.layers.BatchNormalization()
        self.conv0 = conv_1x1(channel)
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.layer0 = self.make_layer(sa_type, block, channel, layers[0], kernels[0])
        
        channel *= 4
        self.conv1 = conv_1x1(channel)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.layer1 = self.make_layer(sa_type, block, channel, layers[1], kernels[1])

        channel *= 2
        self.conv2 = conv_1x1(channel)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.layer2 = self.make_layer(sa_type, block, channel, layers[2], kernels[2])

        channel *= 2
        self.conv3 = conv_1x1(channel)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.layer3 = self.make_layer(sa_type, block, channel, layers[3], kernels[3])

        channel *= 2
        self.conv4 = conv_1x1(channel)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.layer4 = self.make_layer(sa_type, block, channel, layers[4], kernels[4])

        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2, data_format="channels_first")
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first")
        self.fc = tf.keras.layers.Dense(NUM_CLASSES)
    
    def make_layer(self, sa_type, block, planes, num_blocks, kernel_size=7, strides=1):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(block(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, strides))
        return tf.keras.Sequential(layers)

    def call(self, input_):
        x = self.relu(self.initial_bn(self.inital_conv(input_)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.maxpool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.maxpool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.maxpool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.maxpool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.maxpool(x)))))
        x = self.avgpool(x)
        x = self.fc(x)
        return x

def san(sa_type, layers, kernels):
    model = SAN(sa_type, Transition, layers, kernels)
    return model 